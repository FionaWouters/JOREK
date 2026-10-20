!> Particle evolution for the 'rep' coupling scheme: runaway electrons.
!> Called from evolve_particle_group (mod_particle_evolution.f90).
module mod_runaway_evolution
    use particle_tracer
    use coupling_variables
    use mod_ccoll_relativistic, only: ccoll_data, ccoll_init
    use phys_module, only: use_manual_random_seed, part_group_configs, imp_cor
    use, intrinsic :: iso_c_binding, only: c_int64_t
    !$ use omp_lib

    implicit none
    private
    public :: evolve_REs, re_ccoll_data !< public for regtesting, used by evolve_particle_group

    !> Spacing between the generator stream ids of two MPI ranks.
    !>
    !> A rank seeds one stream per particle at stream_base + particle index, so
    !> the ranks must not overlap. A fixed stride rather than the particle count
    !> keeps a rank's ids where they were when the count changes between fluid
    !> steps, and 1e12 is far more slots than any rank will ever hold.
    integer(c_int64_t), parameter :: rng_rank_stride = 1000000000000_c_int64_t

    !> The collision table, read once per run rather than once per call.
    !>
    !> ccoll_init reads an HDF5 file and builds the ion data; nothing in it
    !> depends on the step, so re-reading it every fluid step would be pure I/O.
    !> Kept here rather than in the sim object because it is a property of the
    !> plasma composition, which the sim does not carry.
    type(ccoll_data), save, target :: re_ccoll_data   !< target: c_loc takes its four tables
    logical,          save :: re_ccoll_ready = .false.

    !> Interface only -- the bodies are in C++. One signature, because the two
    !> entry points are one kernel; which one is bound is a build decision.
    abstract interface
      subroutine jgx_evolve_REs_iface(part_base, n_particles, &
                                      el_base, n_elements,    &
                                      nd_base, n_nodes,       &
                                      interp_base,            &
                                      rhs_data, rhs_ext,      &
                                      i_P_par, i_P_perp,      &
                                      i_j_Phi,                &
                                      mass, time, timestep,   &
                                      nstep, phi_search,      &
                                      use_ccoll, use_radreact,&
                                      ccoll_nu, ccoll_nth,    &
                                      ccoll_u, ccoll_theta,   &
                                      ccoll_L0, ccoll_L1,     &
                                      ccoll_n_ions,           &
                                      ccoll_mi, ccoll_Z0,     &
                                      ccoll_m_i_over_m_imp,   &
                                      var_rho, var_T,         &
                                      var_Ti, var_Te,         &
                                      var_rhoimp,             &
                                      cor_n_d, cor_n_T,       &
                                      cor_n_Z, cor_density,   &
                                      cor_temperature,        &
                                      cor_Z,                  &
                                      rng_seed, rng_stream_base, &
                                      not_found, nf_R, nf_Z,  & !> debug
                                      bad_i_from, bad_i_to) bind(C)
        use, intrinsic :: iso_c_binding, only: c_double, c_int32_t, c_int64_t, c_ptr
        implicit none
        type(c_ptr),        value, intent(in)  :: part_base, el_base, nd_base, interp_base
        integer(c_int32_t), value, intent(in)  :: n_particles, n_elements, n_nodes
        real(c_double),            intent(inout) :: rhs_data(*)
        integer(c_int32_t),        intent(in)  :: rhs_ext(5)
        integer(c_int32_t), value, intent(in)  :: i_P_par, i_P_perp, i_j_Phi, nstep
        real(c_double),     value, intent(in)  :: mass, time, timestep, phi_search
        integer(c_int32_t), value, intent(in)  :: use_ccoll, use_radreact
        integer(c_int32_t), value, intent(in)  :: ccoll_nu, ccoll_nth
        type(c_ptr),        value, intent(in)  :: ccoll_u, ccoll_theta, ccoll_L0, ccoll_L1
        integer(c_int32_t), value, intent(in)  :: ccoll_n_ions          ! NEW
        type(c_ptr),        value, intent(in)  :: ccoll_mi, ccoll_Z0
        real(c_double),     value, intent(in)  :: ccoll_m_i_over_m_imp  ! NEW
        integer(c_int32_t), value, intent(in) :: var_rho
        integer(c_int32_t), value, intent(in) :: var_T
        integer(c_int32_t), value, intent(in) :: var_Ti
        integer(c_int32_t), value, intent(in) :: var_Te
        integer(c_int32_t), value, intent(in) :: var_rhoimp
        integer(c_int32_t), value, intent(in)  :: cor_n_d, cor_n_T, cor_n_Z
        type(c_ptr),        value, intent(in)  :: cor_density, cor_temperature, cor_Z
        integer(c_int64_t), value, intent(in)  :: rng_seed, rng_stream_base
        integer(c_int32_t),        intent(out) :: not_found, bad_i_from, bad_i_to
        real(c_double),            intent(out) :: nf_R, nf_Z
      end subroutine
    end interface

    procedure(jgx_evolve_REs_iface), &
      bind(C, name="jgx_host_runaway_evolution_evolve_REs")     :: evolve_REs_host
#ifdef JGX_HAS_DEVICE
    procedure(jgx_evolve_REs_iface), &
      bind(C, name="jgx_host_runaway_evolution_evolve_REs_device") :: evolve_REs_device
#endif
contains

  !> Gathers the runaway-electron projections of a particle group and pushes its
  !> particles.
  !>
  !> Facade only -- the body is in C++ (runaway_evolution.h next door).
  subroutine evolve_REs(sim, group_num, feedback_rhs, tstep_part_adj, nstep_part_adj)
    use mpi
    use mod_fields_linear, only: jorek_fields_interp_linear
    use mod_find_rz_nearby, only: find_RZ_nearby_phi_search, find_RZ_nearby_report
    use mod_particle_types, only: particle_kinetic_relativistic
    use mod_random_seed, only: random_seed
    use mod_model_settings, only: var_rho, var_T, var_Ti, var_Te, var_rhoimp     ! NEW
    use, intrinsic :: iso_c_binding, only: c_double, c_int32_t, c_int64_t, c_loc, c_null_ptr, c_ptr

    implicit none
    class(particle_sim), target, intent(inout) :: sim
    integer, intent(in)                        :: group_num
    real*8, allocatable, intent(inout)         :: feedback_rhs(:,:,:,:,:)
    real*8,  intent(in)                        :: tstep_part_adj
    integer, intent(in)                        :: nstep_part_adj

    integer(c_int32_t) :: rhs_ext(5), not_found, bad_i_from, bad_i_to
    integer(c_int32_t) :: c_use_ccoll, c_use_radreact, ccoll_nu, ccoll_nth
    type(c_ptr)        :: ccoll_u, ccoll_theta, ccoll_L0, ccoll_L1
    real(c_double)     :: ccoll_mi, ccoll_Z0, nf_R, nf_Z
    integer(c_int64_t) :: rng_seed
    integer            :: i, seed, ierr
    logical            :: use_ccoll, use_radreact
    integer(c_int32_t) :: ccoll_n_ions
    type(c_ptr)        :: ccoll_mi_arr, ccoll_Z0_arr
    real(c_double)     :: ccoll_m_i_over_m_imp
    integer(c_int32_t) :: c_var_rho, c_var_T, c_var_Ti, c_var_Te, c_var_rhoimp
    real(c_double), allocatable, target, save :: ccoll_Z0_real(:)
    integer(c_int32_t) :: cor_n_d, cor_n_T, cor_n_Z
    type(c_ptr)        :: cor_density, cor_temperature, cor_Z
    procedure(jgx_evolve_REs_iface), pointer :: evolve_REs_kernel

    real(c_double), allocatable, target, save :: cor_density_copy(:)
    real(c_double), allocatable, target, save :: cor_temperature_copy(:)
    real(c_double), allocatable, target, save :: cor_Z_copy(:,:,:)

#if defined(fullmhd) || STELLARATOR_MODEL
    error stop "evolve_REs: the C++ kernel is reduced MHD only (calc_EBpsiU_reduced)"
#else
    ! One kernel, two entry points: a device build runs on the device, a host
    ! build over the Fortran storage in place. Decided by the build, not by input.
#ifdef JGX_HAS_DEVICE
    evolve_REs_kernel => evolve_REs_device
#else
    evolve_REs_kernel => evolve_REs_host
#endif

    use_ccoll    = part_group_configs(group_num)%use_ccoll
    use_radreact = part_group_configs(group_num)%use_radreact

    ! The collision operator covers one background electron species and one main
    ! ion species at the same temperature -- the with_impurities = .false.,
    ! with_TiTe = .false. branch of calc_NjTj. The partial-screening operator is
    ! not ported at all. Guarded here rather than in the kernel, which cannot
    ! report anything.
    if (use_ccoll) then
!#if defined(WITH_TiTe) || defined(WITH_Impurities)
!      error stop "evolve_REs: use_ccoll needs with_TiTe = .f. and with_impurities = .f."
!#endif
      if (.not. re_ccoll_ready) then
        call ccoll_init('ccolldata', re_ccoll_data)
        re_ccoll_ready = .true.
      end if
!      if (size(re_ccoll_data%mi) /= 1) &
!        error stop "evolve_REs: the C++ collision operator takes exactly one ion species"
    end if

    ! One seed for the whole run, drawn on rank 0 and broadcast, as every other
    ! generator in JOREK is set up. The stream itself is chosen per particle
    ! inside the kernel -- see jorek::re_seed_stream.
    rng_seed = 0_c_int64_t
    if (use_ccoll) then
      if (sim%my_id == 0) seed = random_seed()
      call MPI_Bcast(seed, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)
      rng_seed = int(seed, c_int64_t)
    end if

    do i = 1, 5
      rhs_ext(i) = int(size(feedback_rhs, i), c_int32_t)
    end do

    ! The table crosses as its four arrays with their two extents, and the ion
    ! data as two scalars: ccoll_data holds allocatables, so it is not a record
    ! the registry could describe.
    ccoll_nu    = 0_c_int32_t
    ccoll_nth   = 0_c_int32_t
    ccoll_u     = c_null_ptr
    ccoll_theta = c_null_ptr
    ccoll_L0    = c_null_ptr
    ccoll_L1    = c_null_ptr
    ccoll_mi    = 0.d0
    ccoll_Z0    = 0.d0
    if (use_ccoll) then
      ccoll_nu    = int(size(re_ccoll_data%u),     c_int32_t)
      ccoll_nth   = int(size(re_ccoll_data%theta), c_int32_t)
      ccoll_u     = c_loc(re_ccoll_data%u(1))
      ccoll_theta = c_loc(re_ccoll_data%theta(1))
      ccoll_L0    = c_loc(re_ccoll_data%L0(1,1))
      ccoll_L1    = c_loc(re_ccoll_data%L1(1,1))
      !ccoll_mi    = re_ccoll_data%mi(1)
      !ccoll_Z0    = real(re_ccoll_data%Z0(1), 8)   ! integer*1 in ccoll_data
      ccoll_n_ions        = int(size(re_ccoll_data%mi), c_int32_t)
      ccoll_mi_arr        = c_loc(re_ccoll_data%mi(1))
      if (.not. allocated(ccoll_Z0_real) .or. &
          size(ccoll_Z0_real) /= size(re_ccoll_data%Z0)) then
        if (allocated(ccoll_Z0_real)) deallocate(ccoll_Z0_real)
        allocate(ccoll_Z0_real(size(re_ccoll_data%Z0)))
      end if
      ccoll_Z0_real = real(re_ccoll_data%Z0, 8)
      ccoll_Z0_arr  = c_loc(ccoll_Z0_real(1))
      ccoll_m_i_over_m_imp = re_ccoll_data%m_i_over_m_imp

      c_var_rho    = int(var_rho - 1, c_int32_t)  ! always active, always var_rho-1

      ! For optional variables: Fortran uses 0 to mean "not active",
      ! convert to -1 for C++:
      c_var_T      = merge(int(var_T      - 1, c_int32_t), -1_c_int32_t, var_T      > 0)
      c_var_Ti     = merge(int(var_Ti     - 1, c_int32_t), -1_c_int32_t, var_Ti     > 0)
      c_var_Te     = merge(int(var_Te     - 1, c_int32_t), -1_c_int32_t, var_Te     > 0)
      c_var_rhoimp = merge(int(var_rhoimp - 1, c_int32_t), -1_c_int32_t, var_rhoimp > 0)
    end if

    cor_n_d         = 0_c_int32_t
    cor_n_T         = 0_c_int32_t
    cor_n_Z         = 0_c_int32_t
    cor_density     = c_null_ptr
    cor_temperature = c_null_ptr
    cor_Z           = c_null_ptr

    if (use_ccoll .and. size(re_ccoll_data%mi) > 1) then
      ! Guard against species with too many charge states for the C++ fixed arrays.
      ! MAX_CHARGE_STATES = 19 in coronal.h, covers Ne (n_Z=10) and Ar (n_Z=18).
      if (imp_cor(1)%n_Z > 19) &
        error stop "evolve_REs: impurity atomic number > 19 not supported in &
                  &GPU collision operator (covers Ne and Ar only)"

      ! Check the coronal table is actually allocated -- it is filled by
      ! init_imp_adas, which must be called before evolve_REs.
      if (.not. allocated(imp_cor(1)%density)) &
        error stop "evolve_REs: imp_cor(1)%density not allocated -- &
                  &was init_imp_adas called?"
      if (.not. allocated(imp_cor(1)%temperature)) &
        error stop "evolve_REs: imp_cor(1)%temperature not allocated -- &
                  &was init_imp_adas called?"
      if (.not. allocated(imp_cor(1)%Z)) &
        error stop "evolve_REs: imp_cor(1)%Z not allocated -- &
                  &was init_imp_adas called?"

      ! Pack the three arrays as bare pointers with their extents.
      ! The C++ side reconstructs jgx::views from these.
      ! imp_cor(1)%Z is declared as Z(n_d, n_T, 0:n_Z) in the coronal type,
      ! so c_loc of element (1,1,0) gives the address of the first element.
      cor_n_d         = int(size(imp_cor(1)%density),     c_int32_t)
      cor_n_T         = int(size(imp_cor(1)%temperature), c_int32_t)
      cor_n_Z         = int(imp_cor(1)%n_Z,               c_int32_t)

      ! Copy into target arrays so c_loc can take their address.
      ! This is the same pattern used for ccoll_Z0_real above.
      ! The save attribute keeps the memory alive after this subroutine returns,
      ! which is necessary because the C++ kernel runs asynchronously on the GPU.
      if (.not. allocated(cor_density_copy) .or. &
          size(cor_density_copy) /= cor_n_d) then
        if (allocated(cor_density_copy)) deallocate(cor_density_copy)
        allocate(cor_density_copy(cor_n_d))
      end if
      cor_density_copy = imp_cor(1)%density

      if (.not. allocated(cor_temperature_copy) .or. &
          size(cor_temperature_copy) /= cor_n_T) then
        if (allocated(cor_temperature_copy)) deallocate(cor_temperature_copy)
        allocate(cor_temperature_copy(cor_n_T))
      end if
      cor_temperature_copy = imp_cor(1)%temperature

      ! Z has shape (n_d, n_T, 0:n_Z) -- copy the whole 3D array.
      ! The third dimension runs from 0 to n_Z inclusive = n_Z+1 elements.
      if (.not. allocated(cor_Z_copy) .or. &
          size(cor_Z_copy, 1) /= cor_n_d .or. &
          size(cor_Z_copy, 2) /= cor_n_T .or. &
          size(cor_Z_copy, 3) /= cor_n_Z + 1) then
        if (allocated(cor_Z_copy)) deallocate(cor_Z_copy)
        allocate(cor_Z_copy(cor_n_d, cor_n_T, 0:cor_n_Z))
      end if
      cor_Z_copy = imp_cor(1)%Z

      cor_density     = c_loc(cor_density_copy(1))
      cor_temperature = c_loc(cor_temperature_copy(1))
      cor_Z           = c_loc(cor_Z_copy(1,1,0))
    end if

    ! Logicals cross as int32 and are tested against zero on the other side.
    c_use_ccoll    = merge(1_c_int32_t, 0_c_int32_t, use_ccoll)
    c_use_radreact = merge(1_c_int32_t, 0_c_int32_t, use_radreact)

    select type (particles => sim%groups(group_num)%particles)
    type is (particle_kinetic_relativistic)
      if (size(particles,1) <= 0) return

      if(use_manual_random_seed) then
        !$ call omp_set_schedule(omp_sched_static,10)
      else
        !$ call omp_set_schedule(omp_sched_dynamic,10)
      end if

      write(*,*) 'just before entering C++ thingies'
      select type (fi => sim%fields%interp)
      type is (jorek_fields_interp_linear)
        call evolve_REs_kernel(                                                 &
               c_loc(particles(1)), int(size(particles,1), c_int32_t),          &
               c_loc(sim%fields%element_list%element(1)),                       &
               int(sim%fields%element_list%n_elements, c_int32_t),              &
               c_loc(sim%fields%node_list%node(1)),                             &
               int(sim%fields%node_list%n_nodes, c_int32_t),                    &
               c_loc(fi),                                                       &
               feedback_rhs, rhs_ext,                                           &
               int(P_par_idx_kin  - 1, c_int32_t),                              &
               int(P_perp_idx_kin - 1, c_int32_t),                              &
               int(j_Phi_idx_kin  - 1, c_int32_t),                              &
               sim%groups(group_num)%mass, sim%time, tstep_part_adj,            &
               int(nstep_part_adj, c_int32_t),                                  &
               find_RZ_nearby_phi_search(0.d0),                                 &
               c_use_ccoll, c_use_radreact,                                     &
               ccoll_nu, ccoll_nth,                                             &
               ccoll_u, ccoll_theta, ccoll_L0, ccoll_L1,                        &
               ccoll_n_ions, ccoll_mi_arr, ccoll_Z0_arr, ccoll_m_i_over_m_imp,  &
               c_var_rho, c_var_T, c_var_Ti, c_var_Te, c_var_rhoimp,            &
!               ccoll_mi, ccoll_Z0,                                              &
               cor_n_d, cor_n_T, cor_n_Z,                                       &
               cor_density, cor_temperature, cor_Z,                         &
               rng_seed, int(sim%my_id, c_int64_t)*rng_rank_stride,             &
               not_found, nf_R, nf_Z, bad_i_from, bad_i_to)

        ! The kernel cannot print; find_RZ_nearby's diagnostics come back instead.
        call find_RZ_nearby_report(int(not_found), int(bad_i_from), int(bad_i_to), nf_R, nf_Z)
      class default
        error stop "evolve_REs: the C++ kernel needs jorek_fields_interp_linear"
      end select
    end select
#endif
  end subroutine evolve_REs

end module mod_runaway_evolution
