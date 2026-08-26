!> Runaway-electron avalanche module: knock-on (Moller) collisions and marker resampling.
!>
!> The physics follows the standard secondary-generation model:
!>  - Moller_scatt:    large-angle electron-electron (knock-on) collisions. Each RE marker
!>                     spawns two child markers (scattered + secondary) with the collision
!>                     probability absorbed into the weights, so the marker count triples.
!>  - do_binning_mpi:  resampling of f(X, p_para, mu) on a spatial grid (JOREK elements,
!>                     optionally subdivided in (s,t), equidistant toroidal bins) with
!>                     equidistant momentum-space bins per volume bin. Called when the
!>                     marker count approaches n_limit/3.
!>
!> This is an optimized rewrite of the original implementation. Subroutine names and the
!> order of the code blocks are unchanged; the main differences are:
!>  - group_num is passed explicitly everywhere (the group index and the array size n_limit
!>    were swapped in the old reorder_indices calls, and sim%groups(1) was hard-coded).
!>  - All debug MPI_ALLREDUCE/MPI_BCAST calls inside the per-bin resampling loops are gone;
!>    the rank assignment of new markers now uses a shared rng (same seed and stream on all
!>    ranks) instead of one MPI_BCAST per marker.
!>  - Loop-invariant quantities (classical electron radius, rejection-sampling envelope,
!>    bin indices) are computed once instead of per particle / per iteration.
!>  - p_gc_array is allocatable instead of an automatic (stack) array.
!>  - The momentum-bin histogram lost its unused third component (3x less memory) and has
!>    the bin index first for contiguous access.
!>  - Random numbers: same generators and seeds, but the call sequence differs slightly,
!>    so results are statistically equivalent, not bit-identical, to the old version.
module mod_re_avalanche
    use particle_tracer
    use mod_particle_io
    use mod_particle_sim
    use mod_particle_diagnostics
    use mod_fields_linear
    use mod_fields_hermite_birkhoff
    use mod_kinetic_relativistic
    use mod_gc_relativistic
    use mod_sampling
    use mod_basisfunctions
    use mod_random_seed
    use mod_ccoll_relativistic
    use mod_pusher_tools
    use mpi
    !$ use omp_lib
    implicit none
    private

    public :: reorder_indices, Moller_scatt, do_binning_mpi, type_re_avalanche, &
              re_avalanche_from_config, gcd_re_avalanche!, initialize

    type :: type_re_avalanche
      integer :: group_num
      integer*4 :: element_part_s, element_part_t, Ntor, n_limit
      integer*4, allocatable :: emptyind(:)
      integer :: each_nstep_part
      integer*4 :: Nmombin, new_markers_per_element
      real*8 :: gamma_min
      logical :: constructed=.false.
    contains
      procedure :: initialize
      procedure :: do => do_RE_avalanche
    end type

    integer :: gcd_re_avalanche = -9999991

contains

  !> Subroutine to reorder the particle indices.
  !> Let N_tot be the total amount of markers and N_active the amount of markers with i_elm>0.
  !> Output is the array emptyind which holds the indices of all active markers in
  !> emptyind(1:N_active) and of all inactive markers in emptyind((N_active+1):N_tot).
  !>
  !> Same two-pointer partitioning as before, written as an explicit swap loop.
  !> NOTE: the second argument used to receive n_part_bound while being used as a group
  !> index (and group 1 was hard-coded in the loop body); it is now really the group number.
  subroutine reorder_indices(sim, group_num, emptyind)
    type(particle_sim)       :: sim
    integer,   intent(in)    :: group_num
    integer*4, intent(inout) :: emptyind(:)

    integer*4 :: n_part_bound, i_empty, i_full, storedindex
    real*8 :: start_time, end_time, tot_time(3)

    n_part_bound = size(sim%groups(group_num)%particles(:))

    start_time = MPI_WTIME()

    i_empty = 1
    i_full  = n_part_bound

    do while (i_empty .lt. i_full)

        ! advance i_empty to the next inactive marker (searching to the right)
        do while (i_empty .lt. i_full)
            if (sim%groups(group_num)%particles(emptyind(i_empty))%i_elm .le. 0) exit
            i_empty = i_empty + 1
        end do

        ! retreat i_full to the next active marker (searching to the left)
        do while (i_full .gt. i_empty)
            if (sim%groups(group_num)%particles(emptyind(i_full))%i_elm .gt. 0) exit
            i_full = i_full - 1
        end do

        if (i_empty .ge. i_full) exit ! inactive spot right of active spot: sorting finished

        ! swap the two index entries
        storedindex       = emptyind(i_full)
        emptyind(i_full)  = emptyind(i_empty)
        emptyind(i_empty) = storedindex

        i_empty = i_empty + 1
        i_full  = i_full - 1

    end do

    end_time = MPI_WTIME()
    tot_time = mpi_minmeanmax(end_time-start_time)
    if (sim%my_id .eq. 0) then
        write(*,"(A,3f10.3,A)") , "Time taken for reordering finished in (min/mean/max): ", tot_time, " seconds"
    endif
  end subroutine reorder_indices

  !> Knock-on (large-angle) collisions based on the Moller cross-section.
  !> Every active RE marker with gamma_0 > gamma_min spawns two child markers with weight
  !> prob*w while its own weight becomes (1-prob)*w; energies/angles follow the differential
  !> cross-section (rejection sampling for gamma, then theta/phi from conservation).
  !>
  !> Signature change w.r.t. the old version: group_num added; n_part_bound and n_part_old
  !> dropped (the caller passed n_part for both n_part and n_part_old, which aliased an
  !> inout with an in argument; n_part_old is now simply set to n_part on entry).
  subroutine Moller_scatt(sim, group_num, gamma_min, dt_coll, n_part, emptyind)
    use constants, only: pi, eps_zero, el_chg, c_light, mass_electron

    type(particle_sim)       :: sim
    integer,   intent(in)    :: group_num   !< particle group of the runaway electrons
    real*8,    intent(in)    :: gamma_min   !< cut-off energy of the secondaries as Lorentz factor
    real*8,    intent(in)    :: dt_coll     !< time between knock-on collisions [s]
    integer*4, intent(inout) :: n_part      !< amount of active markers, updated here (+2 per collision)
    integer*4, intent(inout) :: emptyind(:) !< active marker indices first, then inactive (cf. reorder_indices)

    type(pcg32_rng), allocatable :: rng1(:), rng2(:), rng3(:)   ! rngs used for the collisions

    real*8 :: gamma_0                                 ! Lorentz factor incoming RE
    real*8 :: gamma, gamma_prime                      ! Lorentz factors (energies) of the collision products
    real*8 :: theta, phi                              ! Angles related to the collision
    real*8 :: sigma                                   ! Total cross-section
    real*8 :: prob                                    ! Probability of a collision
    real*8 :: re                                      ! Classical electron radius
    real*8 :: ne                                      ! Free electron density
    real*8 :: mass                                    ! Marker (electron) mass
    real*8 :: p2, v0                                  ! |p|^2 and |v| of the incoming RE
    real*8 :: S_max                                   ! Envelope of the rejection sampling
    real*8 :: rand, rand_rng1(1), rand_rng2(1), rand_rng3(1)
    real*8 :: pnorm(3), vector(3), n1(3), n2(3)       ! Direction of incoming momentum and orthonormal vectors
    real*8 :: DUMMY_REAL
    integer*4 :: j, i_child, n_part_old
    integer*4 :: process_rank, amount_processes, ierror, ifail
    integer*4 :: seed1, seed2, seed3
    integer*4 :: n_threads, n_streams, seq, i_thread, i_rng

    call reorder_indices(sim, group_num, emptyind)
    n_part_old = n_part

    call MPI_COMM_RANK(MPI_COMM_WORLD, process_rank, ierror)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, amount_processes, ierror)

    ! Initialize the needed rngs (scalars now; they were size-1 allocatable arrays)
    !call rng1%initialize(n_dims=1, seed=random_seed(), n_streams=amount_processes, i_stream=process_rank+1, ierr=ifail)
    !call rng2%initialize(n_dims=1, seed=random_seed(), n_streams=amount_processes, i_stream=process_rank+1, ierr=ifail)
    !call rng3%initialize(n_dims=1, seed=random_seed(), n_streams=amount_processes, i_stream=process_rank+1, ierr=ifail)

    ! Calculate 1 seed and communicate over MPI
    if (sim%my_id .eq. 0) seed1 = random_seed()
    if (sim%my_id .eq. 0) seed2 = random_seed()
    if (sim%my_id .eq. 0) seed3 = random_seed()
    call MPI_Bcast(seed1, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierror)
    call MPI_Bcast(seed2, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierror)
    call MPI_Bcast(seed3, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierror)
    ! Set up actual rng
    n_threads = 1
    !$ n_threads = omp_get_max_threads()
    allocate(rng1(0:n_threads-1))
    allocate(rng2(0:n_threads-1))
    allocate(rng3(0:n_threads-1))
    n_streams = sim%n_mpi*n_threads
    do i_thread = 0, n_threads-1 
      seq = sim%my_id*n_threads + i_thread + 1
      call rng1(i_thread)%initialize(1, seed1, n_streams, seq, ierror)
      call rng2(i_thread)%initialize(1, seed2, n_streams, seq, ierror)
      call rng3(i_thread)%initialize(1, seed3, n_streams, seq, ierror)
      if(ierror .ne. 0) then 
        call MPI_ABORT(MPI_COMM_WORLD, -1, ierror)
        write(*,*) "WARNING: Set-up rng's in large-angle (knock-on) collisions failed."
      end if 
    end do 

    ! Loop invariants (were recomputed for every particle; re now uses the MASS_ELECTRON
    ! constant instead of the hard-coded 9.109d-31)
    re   = el_chg**2.d0/(4.d0*pi*eps_zero*mass_electron*c_light**2.d0) ! Classical electron radius
    mass = sim%groups(group_num)%mass

    select type (parts => sim%groups(group_num)%particles)
    type is (particle_kinetic_relativistic)

    !$omp parallel default(none) &
    !$omp shared(n_part_old, DUMMY_REAL, emptyind, sim, re, mass, dt_coll, gamma_min) &
    !$omp shared(rng1, rng2, rng3) &
    !$omp private(gamma_0, gamma, gamma_prime, theta, phi, sigma, prob, p2) &
    !$omp private(v0, S_max, rand, rand_rng1, rand_rng2, rand_rng3, pnorm, vector) &
    !$omp private(n1, n2, ne, i_child) &
    !$omp private(j, i_rng) &
    !$omp reduction(+:n_part) 

    i_rng = 1
    !$ i_rng = omp_get_thread_num()

    ! Loop over all particles
    !$omp do schedule(dynamic,64)
    do j=1,n_part_old

      associate(pini => parts(emptyind(j)))

      if (pini%i_elm .le. 0) cycle

      call sim%fields%calc_NeTe(sim%time, pini%i_elm, pini%st, pini%x(3), ne, DUMMY_REAL)
      if (ne .le. 0.d0) cycle

      p2      = dot_product(pini%p, pini%p)
      gamma_0 = sqrt(p2/(mass*c_light)**2.d0 + 1.d0)

      if (gamma_0 .le. gamma_min) cycle ! Large-angle collision is impossible
      if (0.5d0*(gamma_0 + 1.d0) .lt. gamma_min) cycle ! Large-angle collision would lead to negative growthrate (comment this line if you want to include this)

      v0 = c_light*sqrt(gamma_0**2.d0 - 1.d0)/gamma_0 ! |v| of the incoming RE (equals the old sqrt(dot_product(vel,vel)))

      sigma = (4.d0*pi*re**2.d0/(gamma_0**2.d0-1.d0))*(gamma_0**2.d0*(gamma_0-2.d0*gamma_min+1.d0)/((gamma_min-1.d0)*(gamma_0-gamma_min)) &
              + 0.5d0*(gamma_0+1.d0)- gamma_min -(2.d0*gamma_0-1.d0)/(gamma_0-1.d0)*log((gamma_0-gamma_min)/(gamma_min-1.d0)))
      prob = sigma*ne*v0*dt_coll/2.d0 ! Probability of the RE having collided during dt_coll

      if ((prob .lt. 0.d0) .or. (prob .gt. 1.d0)) then
        !$omp critical
        write(*,*) 'WARNING: timestep dt_coll chosen too large, current dt_coll:', dt_coll, 'Leads to collisional probability:', prob, 'Decrease dt_coll to at most', 1/(2*prob/dt_coll)
        !$omp end critical
      end if

      ! Select the outgoing energy gamma of one of the electrons by rejection sampling of the
      ! probability density proportional to the differential cross-section dsigma/dgamma.
      ! The old version divided both sides of the acceptance test by the same constant
      ! sigma/(2*pi*re^2/(gamma_0^2-1)); that factor cancels, so we only compare the shape
      ! function moller_S (identical acceptance decisions, envelope S_max = moller_S(gamma_min)).
      S_max = moller_S(gamma_min, gamma_0)
      do
        call rng1(i_rng)%next(rand_rng1)
        call rng2(i_rng)%next(rand_rng2)
        gamma = rand_rng2(1)*(1.d0 + gamma_0 - 2.d0*gamma_min) + gamma_min
        if (rand_rng1(1)*S_max .le. moller_S(gamma, gamma_0)) exit
      end do

      ! Now we know gamma, we can calculate the energies and angles of all particles after the collision
      theta = acos(sqrt((gamma-1.d0)/(gamma+1.d0))*sqrt((gamma_0+1.d0)/(gamma_0-1.d0)))
      gamma_prime = gamma_0 + 1.d0 - gamma
      phi = asin(sqrt((gamma**2.d0-1.d0)/(gamma_prime**2.d0-1.d0))*sin(theta))

      ! The 2D collision plane: direction of the incoming momentum and a random orthonormal vector
      call rng3(i_rng)%next(rand_rng3)
      rand = rand_rng3(1)
      call get_orthonormals(pini%p, n1, n2)
      vector = cos(rand*2.d0*Pi)*n1 + sin(rand*2.d0*Pi)*n2 ! A vector orthonormal to p

      ! Select target and colliding markers (two inactive slots per parent, as before)
      i_child = n_part_old + 1 + (j-1)*2
      associate(pc => parts(emptyind(i_child)), pt => parts(emptyind(i_child+1)))

      pc = pini
      pt = pini

      pc%weight   = prob*pini%weight
      pt%weight   = prob*pini%weight
      pini%weight = (1.d0-prob)*pini%weight

      pnorm = pini%p/sqrt(p2) ! Direction momentum vector

      pt%p = mass*c_light*sqrt(gamma**2.d0-1.d0)*(cos(theta)*pnorm - sin(theta)*vector)
      pc%p = mass*c_light*sqrt(gamma_prime**2.d0-1.d0)*(cos(phi)*pnorm + sin(phi)*vector)

      n_part = n_part+2 ! Update amount of particles

      end associate
      end associate

    end do
    !$omp end do 
    !$omp end parallel

    class default
      if (sim%my_id .eq. 0) write(*,*) 'ERROR: Moller_scatt requires particles of type particle_kinetic_relativistic. Aborting.'
      stop
    end select

  contains

    !> Shape of the differential Moller cross-section dsigma/dgamma (without the constant
    !> prefactor 2*pi*re^2/(gamma_0^2-1), which cancels in the rejection sampling)
    pure function moller_S(g, g0) result(S)
      real*8, intent(in) :: g, g0
      real*8 :: S
      S = (g0/(g-1.d0))**2 + (g0/(g0-g))**2 + 1.d0 &
          - (2.d0*g0-1.d0)/(g0-1.d0)*(1.d0/(g-1.d0) + 1.d0/(g0-g))
    end function moller_S

  end subroutine Moller_scatt

  !> Resampling of the RE markers: reproduce f(X, p_para, mu) with fewer markers.
  !> Phases (matching the blocks of the old version):
  !>   1. convert all markers to guiding-center (iteratively) and record spatial bin counts
  !>      and per-bin min/max of p_para and mu
  !>   2. MPI reduction of counts/bounds, widen the momentum bounds so the outermost bin
  !>      centers sit on the min/max values
  !>   3. build the per-volume-bin marker lists
  !>   4. accumulate the (p_para, mu) weight histogram per volume bin, MPI reduction
  !>   5. place the new markers: single-marker and few-marker bins are copied as-is,
  !>      otherwise new_markers_per_element markers are drawn (momentum bin by inversion
  !>      sampling, position uniform in (R,Z) by rejection sampling on the Jacobian)
  !>
  !> Signature change w.r.t. the old version: group_num added, n_part_bound dropped
  !> (it was only forwarded to reorder_indices, which now derives it itself).
  subroutine do_binning_mpi(sim, group_num, n_part, n_limit, Nbin, Ntor, element_part_s, element_part_t, &
                            new_markers_per_element, emptyind, iterations)
    use constants, only: pi
    use mod_pcg32_rng
    use mod_interp, only: interp_RZ
    !use mpi_f08
    use mod_particle_sorting, only: sort_particles_in_elm, indices_in_elm 

    type(particle_sim)       :: sim
    integer,   intent(in)    :: group_num       !< particle group of the runaway electrons
    integer*4, intent(inout) :: n_part          !< in: amount of active markers; out: amount after resampling
    integer*4, intent(in)    :: n_limit         !< size of the particle array
    integer*4, intent(in)    :: Nbin            !< momentum bins per direction (p_para and mu)
    integer*4, intent(in)    :: Ntor            !< toroidal bins
    integer*4, intent(in)    :: element_part_s, element_part_t !< subdivisions of each element in s and t
    integer*4, intent(in)    :: new_markers_per_element        !< new markers per volume bin
    integer*4, intent(inout) :: emptyind(:)
    integer*4, intent(in)    :: iterations      !< iterations for the FO->GC conversion

    type(particle_gc_relativistic)              :: p_gc
    type(particle_gc_relativistic), allocatable :: p_gc_array(:) ! allocatable now (was an automatic/stack array)

    ! rng:        4 dims: Jacobian acceptance, s, t, phi of the new markers
    ! rng1:       momentum-bin inversion sampling and gyro angles
    ! rng2:       uniform (p_para, mu) within the selected momentum bin
    ! rng_shared: same seed AND stream on every rank, so all ranks draw identical numbers;
    !             replaces the rank-0-draws + MPI_BCAST-per-marker construction
    type(pcg32_rng) :: rng, rng1, rng2, rng_shared

    real*8, allocatable :: p_para_min(:,:,:,:), p_para_max(:,:,:,:), mu_min(:,:,:,:), mu_max(:,:,:,:)
    real*8, allocatable :: dp_para(:,:,:,:), dmu(:,:,:,:)
    real*8, allocatable :: cum_prob(:)
    integer*4, allocatable :: amount_markers_spatial_bin(:,:,:)

    real*8 :: weighttot, rand(1), ran(4), ran_p_mu(2), gyro_angle(1)
    real*8 :: B(3), E(3), psi, U
    real*8 :: R, Z, phi, R_s, R_t, Z_s, Z_t, f, DUMMY_REAL
    real*8 :: Rbox(2), Zbox(2), minR, minZ, maxR, maxZ, A_RZ
    real*8 :: p_para, mu
    real*8 :: gc_st(2), gc_phi
    real*8 :: start_time, end_time, tot_time(3)
    integer, dimension(n_vertex_max) :: vertices
    integer*4 :: g, h, i, j, k, l, m, n, index, pk
    integer*4 :: lin                !< linearized (h,g) sub-element index, h+(g-1)*element_part_s
    integer*4 :: km                 !< marker index within p_gc_array
    integer*4 :: kii, kjj           !< momentum bin indices in p_para and mu
    integer*4 :: gc_i_elm, n_elements, max_markers_bin, owner
    integer*4 :: new_markers_per_element_loop
    integer*4 :: process_rank, amount_processes, ifail

    integer*4, allocatable :: vol_bin_start(:,:,:)   ! (Ntor, n_elements, element_part_s*element_part_t + 1)
    integer*4, allocatable :: vol_bin_idx(:)          ! flat list of p_gc_array indices, grouped by bin

    ! Intermediate: per-bin counts (reuse amount_markers_spatial_bin, already allocated)
    ! sorted_by_elm and pa_in_elm from the sorting routine
    type(indices_in_elm), allocatable :: sorted_by_elm(:)
    integer,              allocatable :: pa_in_elm(:)
    class(particle_base), allocatable, dimension(:) :: pa_polymorphic

    integer*4 :: n_lin, ierr
    integer*4, allocatable :: local_next(:,:)
    integer*4 :: offset

    real*8 :: momentum_hist(Nbin**2)

    call MPI_COMM_RANK(MPI_COMM_WORLD, process_rank, ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, amount_processes, ierr)
    if (sim%my_id .eq. 0) write(*,*) 'Amount processes (compare with jobpart settings)', amount_processes

    n_elements = sim%fields%element_list%n_elements

    ! One rng per rank (scalars; every rank used to allocate and initialize all streams)
    call rng%initialize (n_dims=4, seed=random_seed(), n_streams=amount_processes, i_stream=process_rank+1, ierr=ifail)
    call rng1%initialize(n_dims=1, seed=random_seed(), n_streams=amount_processes, i_stream=process_rank+1, ierr=ifail)
    call rng2%initialize(n_dims=2, seed=random_seed(), n_streams=amount_processes, i_stream=process_rank+1, ierr=ifail)
    call rng_shared%initialize(n_dims=1, seed=5, n_streams=1, i_stream=1, ierr=ifail)

    allocate(p_para_min(Ntor,n_elements,element_part_s,element_part_t))
    allocate(p_para_max(Ntor,n_elements,element_part_s,element_part_t))
    allocate(mu_min(Ntor,n_elements,element_part_s,element_part_t))
    allocate(mu_max(Ntor,n_elements,element_part_s,element_part_t))
    allocate(dp_para(Ntor,n_elements,element_part_s,element_part_t))
    allocate(dmu(Ntor,n_elements,element_part_s,element_part_t))

    p_para_min(:,:,:,:) =  1.d30
    p_para_max(:,:,:,:) = -1.d30
    mu_min(:,:,:,:)     =  1.d30
    mu_max(:,:,:,:)     = -1.d30

    allocate(cum_prob(Nbin**2))

    allocate(amount_markers_spatial_bin(Ntor, n_elements, element_part_s*element_part_t))
    amount_markers_spatial_bin(:,:,:) = 0

    call reorder_indices(sim, group_num, emptyind)

    allocate(p_gc_array(n_part))

    ! --------------------------------------------------------------------------------
    ! Phase 1: convert all markers to GC, bin them spatially and record momentum bounds
    ! --------------------------------------------------------------------------------
    start_time = MPI_WTIME()

    weighttot = 0.d0
    pk = 1
    do k = 1,n_part
      select type (p => sim%groups(group_num)%particles(emptyind(k)))
      type is (particle_kinetic_relativistic)
        weighttot = weighttot + p%weight

        ! iterative FO->GC conversion: B is needed at the GC position, which is only
        ! known after converting, so iterate starting from the FO position
        gc_i_elm = p%i_elm
        gc_st    = p%st
        gc_phi   = p%x(3)
        do i=1,iterations
            if (gc_i_elm .le. 0) exit
            call sim%fields%calc_EBpsiU(sim%time, gc_i_elm, gc_st, gc_phi, E, B, psi, U)
            p_gc = relativistic_kinetic_to_relativistic_gc(sim%fields%node_list, sim%fields%element_list, &
                                                           p, sim%groups(group_num)%mass, B)
            gc_i_elm = p_gc%i_elm
            gc_st    = p_gc%st
            gc_phi   = p_gc%x(3)
        end do

        if (p_gc%i_elm .le. 0) cycle ! GC ended up outside the domain: marker is dropped from the resampling

        p_gc_array(pk) = p_gc
        if (p_gc_array(pk)%x(3) .lt. 0.d0) p_gc_array(pk)%x(3) = p_gc_array(pk)%x(3) + 2.d0*Pi

        call get_spatial_bin(p_gc_array(pk), j, h, g, lin)
        i = p_gc_array(pk)%i_elm

        amount_markers_spatial_bin(j,i,lin) = amount_markers_spatial_bin(j,i,lin) + 1
        p_para_max(j,i,h,g) = max(p_para_max(j,i,h,g), p_gc_array(pk)%p(1))
        p_para_min(j,i,h,g) = min(p_para_min(j,i,h,g), p_gc_array(pk)%p(1))
        mu_max(j,i,h,g)     = max(mu_max(j,i,h,g),     p_gc_array(pk)%p(2))
        mu_min(j,i,h,g)     = min(mu_min(j,i,h,g),     p_gc_array(pk)%p(2))

        if (p_gc_array(pk)%p(2) .le. 0.d0) write(*,*) 'Marker with negative mu', p_gc_array(pk)%p(2)

        pk = pk + 1

      type is (particle_gc_relativistic)
        weighttot = weighttot + p%weight

        p_gc_array(pk) = p
        if (p_gc_array(pk)%x(3) .lt. 0.d0) p_gc_array(pk)%x(3) = p_gc_array(pk)%x(3) + 2.d0*Pi

        call get_spatial_bin(p_gc_array(pk), j, h, g, lin)
        i = p_gc_array(pk)%i_elm

        amount_markers_spatial_bin(j,i,lin) = amount_markers_spatial_bin(j,i,lin) + 1
        p_para_max(j,i,h,g) = max(p_para_max(j,i,h,g), p_gc_array(pk)%p(1))
        p_para_min(j,i,h,g) = min(p_para_min(j,i,h,g), p_gc_array(pk)%p(1))
        mu_max(j,i,h,g)     = max(mu_max(j,i,h,g),     p_gc_array(pk)%p(2))
        mu_min(j,i,h,g)     = min(mu_min(j,i,h,g),     p_gc_array(pk)%p(2))

        if (p_gc_array(pk)%p(2) .le. 0.d0) write(*,*) 'Marker with negative mu', p_gc_array(pk)%p(2)

        pk = pk + 1

      end select 
    end do
    n_part = pk - 1 ! amount of successfully converted markers (was count(p_gc_array%i_elm>0))
    if (sim%my_id .eq. 0) write(*,*) 'weighttot p_gc_array', weighttot

    ! --------------------------------------------------------------------------------
    ! Phase 2: global reduction of the bin counts and momentum bounds
    ! --------------------------------------------------------------------------------
    call MPI_ALLREDUCE(MPI_IN_PLACE, p_para_max, size(p_para_max), MPI_DOUBLE_PRECISION, MPI_MAX, MPI_COMM_WORLD, ierr)
    call MPI_ALLREDUCE(MPI_IN_PLACE, p_para_min, size(p_para_min), MPI_DOUBLE_PRECISION, MPI_MIN, MPI_COMM_WORLD, ierr)
    call MPI_ALLREDUCE(MPI_IN_PLACE, mu_max, size(mu_max), MPI_DOUBLE_PRECISION, MPI_MAX, MPI_COMM_WORLD, ierr)
    call MPI_ALLREDUCE(MPI_IN_PLACE, mu_min, size(mu_min), MPI_DOUBLE_PRECISION, MPI_MIN, MPI_COMM_WORLD, ierr)

    ! Widen the bounds such that the centers of the outermost bins are at the min/max values.
    dp_para = (p_para_max-p_para_min)/real(Nbin-1,8)
    dmu = (mu_max-mu_min)/real(Nbin-1,8)
    p_para_max = p_para_max + 0.5d0*dp_para
    p_para_min = p_para_min - 0.5d0*dp_para
    mu_max = mu_max + 0.5d0*dmu
    mu_min = mu_min - 0.5d0*dmu

    deallocate(dp_para)
    deallocate(dmu)

    end_time = MPI_WTIME()
    tot_time = mpi_minmeanmax(end_time-start_time)
    if (sim%my_id .eq. 0) then
        write(*,"(A,3f10.3,A)") , "Time taken for converting all particles & MPI_ALLREDUCE finished in (min/mean/max): ", tot_time, " seconds"
    endif

    ! --------------------------------------------------------------------------------
    ! Phase 3: build the per-volume-bin lists of (local) marker indices
    ! --------------------------------------------------------------------------------
    start_time = MPI_WTIME()

    allocate(pa_polymorphic(size(p_gc_array)), source=p_gc_array)
    do i = 1, size(p_gc_array)
      pa_polymorphic(i) = p_gc_array(i)
    end do

    call sort_particles_in_elm(pa_polymorphic, n_elements, sorted_by_elm, pa_in_elm)

    allocate(vol_bin_start(Ntor+1, n_elements, element_part_s*element_part_t))
    vol_bin_start(:,:,:) = 0

    n_lin = element_part_s * element_part_t
    offset = 1          ! 1-based
    do lin = 1, n_lin
      do i = 1, n_elements
        do j = 1, Ntor
          vol_bin_start(j, i, lin) = offset
          offset = offset + amount_markers_spatial_bin(j, i, lin)
        end do
        vol_bin_start(Ntor+1, i, lin) = offset
      end do
    end do

    ! Allocate the flat index array (one entry per local marker)
    allocate(vol_bin_idx(n_part))
    vol_bin_idx(:) = 0

    n_lin = element_part_s * element_part_t

    !$omp parallel do default(none)                                       &
    !$omp shared(n_elements, pa_in_elm, sorted_by_elm, p_gc_array,        &
    !$omp        vol_bin_start, vol_bin_idx, n_lin, Ntor, n_part, sim, amount_markers_spatial_bin)                  &
    !$omp private(i, k, j, h, g, lin, local_next, km)                          &
    !$omp schedule(dynamic, 1)
    do i = 1, n_elements
      if (pa_in_elm(i) .eq. 0) cycle

      ! local_next is private to this thread's iteration of element i
      allocate(local_next(Ntor, n_lin))
      local_next(:,:) = 0

      do k = 1, pa_in_elm(i)
          km = sorted_by_elm(i)%pa_ind(k)
          call get_spatial_bin(p_gc_array(km), j, h, g, lin)
          ! position in vol_bin_idx = base offset for (j,i,lin) + how many we've
          ! already inserted from this element (other elements don't share a bin
          ! with this element, so local_next is race-free)
          if ((vol_bin_start(j, i, lin) + local_next(j, lin)) .gt. n_part) then
            write(*,*) 'ERROR: cant write into vol_bin_ix'
            write(*,*) 'ERROR cont.', sim%my_id, i, pa_in_elm(i), amount_markers_spatial_bin(j,i,lin), vol_bin_start(j,i,lin), local_next(j, lin)
          end if 
          vol_bin_idx( vol_bin_start(j, i, lin) + local_next(j, lin) ) = km
          local_next(j, lin) = local_next(j, lin) + 1
      end do

      deallocate(local_next)
    end do
    !$omp end parallel do

    end_time = MPI_WTIME()
    tot_time = mpi_minmeanmax(end_time-start_time)
    if (sim%my_id .eq. 0) then
        write(*,"(A,3f10.3,A)") , "Time taken for volume bining finished in (min/mean/max): ", tot_time, " seconds"
    endif

    if (sim%my_id .eq. 0) write(*,*) 'amount of elements in which there is at least 1 marker', count(amount_markers_spatial_bin .ne. 0)

    call MPI_ALLREDUCE(MPI_IN_PLACE, amount_markers_spatial_bin, size(amount_markers_spatial_bin), MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD, ierr)

    ! --------------------------------------------------------------------------------
    ! Phase 4: place the new markers (overwriting the particle array from index 1)
    ! --------------------------------------------------------------------------------
    start_time = MPI_WTIME()
    index = 1
    do g = 1,element_part_t
      do h = 1,element_part_s
        lin = h + (g-1)*element_part_s
        do i = 1, n_elements
          do j = 1, Ntor


            ! If there are globally no markers in the volume bin, skip the resampling here
            if (amount_markers_spatial_bin(j,i,lin) .eq. 0) cycle

            ! Avoid a degenerate (zero-width) momentum range
            if (p_para_max(j,i,h,g) .eq. p_para_min(j,i,h,g)) then
               p_para_max(j,i,h,g) = p_para_max(j,i,h,g)*1.0001d0
               p_para_min(j,i,h,g) = p_para_min(j,i,h,g)*0.9999d0
            end if
            if (mu_max(j,i,h,g) .eq. mu_min(j,i,h,g)) then
               mu_max(j,i,h,g) = mu_max(j,i,h,g)*1.0001d0
               mu_min(j,i,h,g) = mu_min(j,i,h,g)*0.9999d0
            end if
            if (mu_min(j,i,h,g) .lt. 0.d0) mu_min(j,i,h,g) = 0.d0

            momentum_hist(:) = 0.d0 
            weighttot = 0.d0 

            do k = vol_bin_start(j, i, lin), vol_bin_start(j+1, i, lin) - 1
                km = vol_bin_idx(k)
                if (p_gc_array(km)%weight .lt. 0.d0) then
                  write(*,*) 'There is a particle with negative weight'
                  cycle
                end if

                weighttot = weighttot + p_gc_array(km)%weight

                kii = INT((p_gc_array(km)%p(1) - p_para_min(j,i,h,g))/(p_para_max(j,i,h,g) - p_para_min(j,i,h,g))*Nbin+1)
                kjj = INT((p_gc_array(km)%p(2) - mu_min(j,i,h,g))/(mu_max(j,i,h,g) - mu_min(j,i,h,g))*Nbin+1)

                momentum_hist(kjj+(kii-1)*Nbin) = momentum_hist(kjj+(kii-1)*Nbin) + p_gc_array(km)%weight
            end do

            call MPI_ALLREDUCE(MPI_IN_PLACE, momentum_hist, Nbin**2, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, ierr)
            call MPI_ALLREDUCE(MPI_IN_PLACE, weighttot, 1, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, ierr)



            ! --- Case 1: a single marker (globally) in this volume bin: copy it unchanged.
            ! (The per-bin MPI_ALLREDUCE weight check of the old version was removed here.)
            if (amount_markers_spatial_bin(j,i,lin) .eq. 1) then
              if (vol_bin_start(j+1,i,lin) .gt. vol_bin_start(j,i,lin)) then   ! bin is non-empty
                p_gc = p_gc_array(vol_bin_idx(vol_bin_start(j,i,lin)))
                select type (p => sim%groups(group_num)%particles(index))
                type is (particle_kinetic_relativistic)
                  call sim%fields%calc_EBpsiU(sim%time, i, p_gc%st, p_gc%x(3), E, B, psi, U)
                  p = relativistic_gc_to_relativistic_kinetic(sim%fields%node_list, sim%fields%element_list, &
                                                                        p_gc, sim%groups(group_num)%mass, B, 0.d0)
                  if (p%i_elm .le. 0) write(*,*) 'Single particle has been placed outside domain', p%i_elm, p_gc%i_elm
                type is (particle_gc_relativistic)
                  p = p_gc
                end select 
                index = index+1
              end if
              cycle
            end if

            ! --- Case 2: fewer markers in the bin than would be newly created: copy them all unchanged.
            if (new_markers_per_element .gt. amount_markers_spatial_bin(j,i,lin)) then
              do k = vol_bin_start(j, i, lin), vol_bin_start(j+1, i, lin) - 1
                km = vol_bin_idx(k)
                p_gc = p_gc_array(km)
                select type (p => sim%groups(group_num)%particles(index))
                type is (particle_kinetic_relativistic)
                  call sim%fields%calc_EBpsiU(sim%time, i, p_gc%st, p_gc%x(3), E, B, psi, U)
                  p = relativistic_gc_to_relativistic_kinetic(sim%fields%node_list, sim%fields%element_list, &
                                                                        p_gc, sim%groups(group_num)%mass, B, 0.d0)
                  if (p%i_elm .le. 0) write(*,*) 'Multiple but not resampled particle has been placed outside domain', p%i_elm, p_gc%i_elm
                type is (particle_gc_relativistic)
                  p = p_gc
                end select
                index = index+1
              end do
              cycle
            end if

            ! --- Case 3: actual resampling. First determine how many of the
            ! new_markers_per_element markers this mpi rank creates.
            if (new_markers_per_element .lt. amount_processes) then
              ! randomly assign each new marker to one rank; rng_shared produces identical
              ! numbers on all ranks, so they agree on the assignment without communication
              ! (previously one MPI_BCAST per marker)
              new_markers_per_element_loop = 0
              do n = 1, new_markers_per_element
                call rng_shared%next(rand)
                owner = min(int(rand(1)*real(amount_processes,8)), amount_processes-1)
                if (sim%my_id .eq. owner) new_markers_per_element_loop = new_markers_per_element_loop + 1
              end do
            else
              new_markers_per_element_loop = int(real(new_markers_per_element,8)/real(amount_processes,8))
            end if

            ! Normalize the histogram to probabilities and build the cumulative sum for inversion sampling
            momentum_hist = momentum_hist/weighttot
            cum_prob(1) = momentum_hist(1)
            do m = 2, size(cum_prob)
              cum_prob(m) = cum_prob(m-1) + momentum_hist(m)
            end do
            if (abs(cum_prob(size(cum_prob)) - 1.d0) .gt. 1.d-6) write(*,*) 'last value cum_prob:', cum_prob(size(cum_prob))

            ! Bounding box of the element in (R,Z), used for the uniform-in-(R,Z) placement below
            vertices = sim%fields%element_list%element(i)%vertex(:)
            minR = real(minval(sim%fields%node_list%node(vertices)%x(1,1,1)), 8)
            maxR = real(maxval(sim%fields%node_list%node(vertices)%x(1,1,1)), 8)
            minZ = real(minval(sim%fields%node_list%node(vertices)%x(1,1,2)), 8)
            maxZ = real(maxval(sim%fields%node_list%node(vertices)%x(1,1,2)), 8)
            Rbox = [minR - 0.02d0, maxR + 0.02d0]
            Zbox = [minZ - 0.02d0, maxZ + 0.02d0]

            A_RZ = (Rbox(2) - Rbox(1))*(Zbox(2)-Zbox(1))

            select type (parts => sim%groups(group_num)%particles)
            type is (particle_kinetic_relativistic)

            do n=1, new_markers_per_element_loop
              ! Select momentum bin l by inversion sampling of the cumulative probabilities
              call rng1%next(rand)
              l = minloc(cum_prob, dim=1, mask=((rand(1) .gt. [0.d0, cum_prob(1:(size(cum_prob)-1))]) .and. (rand(1) .le. cum_prob)))

              ! Decode l = kjj + (kii-1)*Nbin into the p_para bin kii and the mu bin kjj, and
              ! draw p_para and mu uniformly within that bin
              call rng2%next(ran_p_mu)
              kii = (l-1)/Nbin +1 
              kjj = modulo(l-1, Nbin) + 1

              p_para = (real(kii,8) - ran_p_mu(1))*(p_para_max(j,i,h,g)-p_para_min(j,i,h,g))/real(Nbin,8) + p_para_min(j,i,h,g)
              mu     = (real(kjj,8) - ran_p_mu(2))*(mu_max(j,i,h,g)-mu_min(j,i,h,g))/real(Nbin,8) + mu_min(j,i,h,g)              

              if (mu .le. 0.d0) write(*,*) 'mu is zero or negative', mu, 'mu_max', mu_max(j,i,h,g), 'mu_min', mu_min(j,i,h,g), 'p_para', p_para

              ! Weight of the new markers: conserves the total weight of the volume bin
              parts(index)%q = -1
              if (new_markers_per_element .le. amount_processes) then
                parts(index)%weight = weighttot/real(new_markers_per_element,8)
              else
                parts(index)%weight = weighttot/(real(new_markers_per_element/amount_processes,8)*real(amount_processes,8))
              end if

              ! Position: uniform in (R,Z) within sub-element (h,g) of element i, by rejection
              ! sampling with the Jacobian of the (s,t)->(R,Z) map
              do
                call rng%next(ran)
                ran(2) = real(h-1,8)/real(element_part_s,8) + ran(2)/real(element_part_s,8)
                ran(3) = real(g-1,8)/real(element_part_t,8) + ran(3)/real(element_part_t,8)
                call interp_RZ(sim%fields%node_list, sim%fields%element_list, i, ran(2), ran(3), R, R_s, R_t, Z, Z_s, Z_t)
                f = abs(R_s*Z_t - R_t*Z_s)/A_RZ
                if (ran(1) .le. f) exit
              end do
              phi = 2.d0*Pi/real(Ntor,8)*ran(4) + 2.d0*Pi/real(Ntor,8)*(real(j,8)-1.d0)

              call rng1%next(gyro_angle)
              gyro_angle(1) = gyro_angle(1)*2.d0*pi

              ! Assemble the new GC marker. The conversion of parts(index) only serves to
              ! initialise all GC components consistently (weight and q were set above);
              ! position and momenta are overwritten right after.
              call sim%fields%calc_EBpsiU(sim%time, i, [ran(2), ran(3)], phi, E, B, psi, U)
              p_gc = relativistic_kinetic_to_relativistic_gc(sim%fields%node_list, sim%fields%element_list, &
                                                             parts(index), sim%groups(group_num)%mass, B)
              p_gc%x(1)  = R
              p_gc%x(2)  = Z
              p_gc%x(3)  = phi
              p_gc%st(1) = ran(2)
              p_gc%st(2) = ran(3)

              call find_RZ(sim%fields%node_list,sim%fields%element_list,p_gc%x(1),p_gc%x(2),DUMMY_REAL,DUMMY_REAL,p_gc%i_elm,p_gc%st(1),p_gc%st(2),ifail)
              if (p_gc%i_elm .ne. i) write(*,*) 'element numbers do not match'

              p_gc%p(1) = p_para
              p_gc%p(2) = mu
              p_gc%q    = -1

              ! Convert the new GC marker to a kinetic (FO) marker.
              call sim%fields%calc_EBpsiU(sim%time, i, p_gc%st, p_gc%x(3), E, B, psi, U)
              parts(index) = relativistic_gc_to_relativistic_kinetic(sim%fields%node_list, sim%fields%element_list, &
                                                                     p_gc, sim%groups(group_num)%mass, B, gyro_angle(1))
              index = index + 1
            end do ! new markers

            type is (particle_gc_relativistic)

            do n=1, new_markers_per_element_loop
              ! Select momentum bin l by inversion sampling of the cumulative probabilities
              call rng1%next(rand)
              l = minloc(cum_prob, dim=1, mask=((rand(1) .gt. [0.d0, cum_prob(1:(size(cum_prob)-1))]) .and. (rand(1) .le. cum_prob)))

              ! Decode l = kjj + (kii-1)*Nbin into the p_para bin kii and the mu bin kjj, and
              ! draw p_para and mu uniformly within that bin
              call rng2%next(ran_p_mu)
              kii = (l-1)/Nbin +1 
              kjj = modulo(l-1, Nbin) + 1

              p_para = (real(kii,8) - ran_p_mu(1))*(p_para_max(j,i,h,g)-p_para_min(j,i,h,g))/real(Nbin,8) + p_para_min(j,i,h,g)
              mu     = (real(kjj,8) - ran_p_mu(2))*(mu_max(j,i,h,g)-mu_min(j,i,h,g))/real(Nbin,8) + mu_min(j,i,h,g)              

              if (mu .le. 0.d0) write(*,*) 'mu is zero or negative', mu, 'mu_max', mu_max(j,i,h,g), 'mu_min', mu_min(j,i,h,g), 'p_para', p_para

              ! Weight of the new markers: conserves the total weight of the volume bin
              parts(index)%q = -1
              if (new_markers_per_element .le. amount_processes) then
                parts(index)%weight = weighttot/real(new_markers_per_element,8)
              else
                parts(index)%weight = weighttot/(real(new_markers_per_element/amount_processes,8)*real(amount_processes,8))
              end if

              ! Position: uniform in (R,Z) within sub-element (h,g) of element i, by rejection
              ! sampling with the Jacobian of the (s,t)->(R,Z) map
              do
                call rng%next(ran)
                ran(2) = real(h-1,8)/real(element_part_s,8) + ran(2)/real(element_part_s,8)
                ran(3) = real(g-1,8)/real(element_part_t,8) + ran(3)/real(element_part_t,8)
                call interp_RZ(sim%fields%node_list, sim%fields%element_list, i, ran(2), ran(3), R, R_s, R_t, Z, Z_s, Z_t)
                f = abs(R_s*Z_t - R_t*Z_s)/A_RZ
                if (ran(1) .le. f) exit
              end do
              phi = 2.d0*Pi/real(Ntor,8)*ran(4) + 2.d0*Pi/real(Ntor,8)*(real(j,8)-1.d0)

              call rng1%next(gyro_angle)
              gyro_angle(1) = gyro_angle(1)*2.d0*pi

              parts(index)%x(1)  = R
              parts(index)%x(2)  = Z
              parts(index)%x(3)  = phi
              parts(index)%st(1) = ran(2)
              parts(index)%st(2) = ran(3)

              call find_RZ(sim%fields%node_list,sim%fields%element_list,parts(index)%x(1),parts(index)%x(2),DUMMY_REAL,DUMMY_REAL,parts(index)%i_elm,parts(index)%st(1),parts(index)%st(2),ifail)
              if (parts(index)%i_elm .ne. i) write(*,*) 'element numbers do not match'

              parts(index)%p(1) = p_para
              parts(index)%p(2) = mu
              parts(index)%q    = -1

              index = index + 1
            end do ! new markers

          end select

          end do
        end do
      end do
    end do

    end_time = MPI_WTIME()
    tot_time = mpi_minmeanmax(end_time-start_time)
    if (sim%my_id .eq. 0) then
        write(*,"(A,3f10.3,A)") , "Time taken for last resampling loop finished in (min/mean/max): ", tot_time, " seconds"
    endif

    ! --------------------------------------------------------------------------------
    ! Finalize: deactivate the remaining slots and reset emptyind
    ! --------------------------------------------------------------------------------
    sim%groups(group_num)%particles(index:n_limit)%i_elm  = 0
    sim%groups(group_num)%particles(index:n_limit)%weight = 0.d0

    n_part = count(sim%groups(group_num)%particles(:)%i_elm .gt. 0)

    if (sim%my_id .eq. 0) write(*,*) 'npart', n_part
    if (sim%my_id .eq. 0) write(*,*) 'Total weight at end do binning', sum(sim%groups(group_num)%particles(:)%weight)

    ! all active markers are now stored contiguously from index 1, so the identity is sorted
    emptyind = (/(l, l=1,n_limit, 1)/)

    deallocate(p_para_min)
    deallocate(p_para_max)
    deallocate(mu_min)
    deallocate(mu_max)
    deallocate(amount_markers_spatial_bin)
    deallocate(p_gc_array)
    deallocate(vol_bin_start)
    deallocate(vol_bin_idx)
    deallocate(sorted_by_elm)
    deallocate(pa_in_elm)

  contains

    !> Spatial bin of a GC marker: toroidal bin jbin, sub-element bins (hbin, gbin) in (s,t)
    !> and the linearized sub-element index linbin (same formulas as inlined before)
    subroutine get_spatial_bin(pg, jbin, hbin, gbin, linbin)
      use constants, only: pi
      type(particle_gc_relativistic), intent(in) :: pg
      integer*4, intent(out) :: jbin, hbin, gbin, linbin

      jbin = int(pg%x(3)/(2.d0*Pi)*real(Ntor,8)) + 1
      if (jbin .eq. (Ntor + 1)) jbin = 1
      hbin = int(pg%st(1)*real(element_part_s,8)) + 1
      if (hbin .eq. (element_part_s + 1)) hbin = element_part_s
      gbin = int(pg%st(2)*real(element_part_t,8)) + 1
      if (gbin .eq. (element_part_t + 1)) gbin = element_part_t
      linbin = hbin + (gbin-1)*element_part_s
    end subroutine get_spatial_bin

  end subroutine do_binning_mpi

  function re_avalanche_from_config(sim) result(re_avalanche)
    use phys_module, only: part_group_configs, n_part_groups_max
    use mod_particle_sim, only: group_num_from_id

    implicit none

    type(particle_sim), intent(inout)                      :: sim
    class(type_re_avalanche), allocatable, dimension(:) :: re_avalanche

    integer :: i, group_num, n_reava_objs, i_reava_obj
    character(len=3) :: id

    n_reava_objs = 0
    do i=1, n_part_groups_max
      id = part_group_configs(i)%id
      if(id == "non") cycle
      if(.not. part_group_configs(i)%use_re_avalanche) then
        if(any(abs(part_group_configs(i)%res_st_bin(:) - 1) .gt. 1.d-10) .or.  (abs(part_group_configs(i)%res_phi_bin - 1) .gt. 1.d-10)) then
          if(sim%my_id.eq.0) write(*,*) 'WARNING: currently resampling only works with re-avalanche, part_group_configs(',i,')%res_st_bin and %res_phi_bin will be ignored as part_group_configs(',i,')%use_re_avalanche=.false.'
        end if
        cycle
      end if
      n_reava_objs = n_reava_objs+1
    end do

    allocate(re_avalanche(n_reava_objs))

    i_reava_obj=0
    do i=1,n_part_groups_max
      id = part_group_configs(i)%id
      if(id == "non") cycle
      if(.not. part_group_configs(i)%use_re_avalanche) cycle
      i_reava_obj = i_reava_obj + 1

      if(part_group_configs(i)%coupling_scheme .ne. "rep") then
        if(sim%my_id .eq. 0) write(*,*) 'ERROR: RE avalanche can only be use with runaway electrons, but part_group_configs(',i,')%use_re_avalanche=.true. while part_group_configs(',i,')%coupling_scheme=',part_group_configs(i)%coupling_scheme,' instead of rep. Aborting.'
        stop
      end if

      if(any(part_group_configs(i)%res_st_bin(:) .lt. 1) .or. (part_group_configs(i)%res_phi_bin .lt. 1)) then
        if(sim%my_id .eq. 0) write(*,*) 'Negative values for amount of bins in st in part_group_configs(',i,')%res_st_bin=',part_group_configs(i)%res_st_bin,' or for toroidal bins in part_group_configs(',i,')%res_phi_bin=',part_group_configs(i)%res_phi_bin,' Please check your input. Aborting.'
        stop
      end if
      group_num = group_num_from_id(sim, id)
      call re_avalanche(i_reava_obj)%initialize(sim,group_num,part_group_configs(i)%res_st_bin,part_group_configs(i)%res_phi_bin, size(sim%groups(group_num)%particles(:)), part_group_configs(i)%gamma_min, part_group_configs(i)%new_markers_per_spatial_bin, part_group_configs(i)%bins_per_mom_direction, part_group_configs(i)%reava_each_nstep_part)
    end do

    if(i_reava_obj .ne. n_reava_objs) then
      if(sim%my_id .eq. 0) write(*,*) 'ERROR in setup for RE avalanche: number of objects to be made is inconsistent?', i_reava_obj, n_reava_objs
      stop
    end if

  end function re_avalanche_from_config


  subroutine initialize(this, sim, group_num, st_bin, phi_bin, n_limit, gamma_min, new_markers, Nmombin, each_nstep_part)
    use mod_math_operators, only: gcd
    implicit none
    class(type_re_avalanche),      intent(inout) :: this
    type(particle_sim),            intent(inout) :: sim
    integer,                       intent(in)    :: group_num
    integer*4,                     intent(in)    :: st_bin(2), phi_bin
    integer,                       intent(in)    :: n_limit
    real*8,                        intent(in)    :: gamma_min
    integer*4,                     intent(in)    :: each_nstep_part, new_markers, Nmombin

    integer :: l

    this%group_num = group_num ! was never set in the old version (this%group_num stayed uninitialized)

    this%element_part_s = st_bin(1)
    this%element_part_t = st_bin(2)
    this%Ntor = phi_bin

    allocate(this%emptyind(n_limit))
    this%emptyind = (/(l, l=1,n_limit, 1)/)
    this%n_limit = n_limit

    this%gamma_min = gamma_min
    this%new_markers_per_element = new_markers
    this%Nmombin = Nmombin
    this%each_nstep_part = each_nstep_part

    if (each_nstep_part .ne. 0) then
      call sim%update_lcm_gcd(each_nstep_part)
      if(gcd_re_avalanche .eq. -9999991) then
        gcd_re_avalanche = each_nstep_part
      else
        gcd_re_avalanche = gcd(gcd_re_avalanche,each_nstep_part)
      endif
    endif

    this%constructed = .true.
  end subroutine initialize

  subroutine do_RE_avalanche(this, sim)
    implicit none

    class(type_re_avalanche), intent(inout) :: this
    type(particle_sim),  intent(inout) :: sim
    type(count_action) :: counter

    integer*4 :: n_part, n_limit, n_part_total
    integer*4 :: iterations=10
    real*8 :: start_time, end_time, tot_time(3)
    real*8 :: dt_coll
    logical :: resample
    integer*4 :: amount_processes, ierr

    call MPI_COMM_SIZE(MPI_COMM_WORLD, amount_processes, ierr)

    if(.not. this%constructed) then
      if(sim%my_id.eq.0) write(*,*) 'ERROR: Something went wrong in initializing RE avalanche. Aborting.'
      stop
    end if

    call reorder_indices(sim, this%group_num, this%emptyind)

    resample = .false. 

    n_part = count(sim%groups(this%group_num)%particles(:)%i_elm .gt. 0)
    !if (sim%my_id .eq. 0) write(*,*) 'Amount living particles, id'
    !write(*,*) n_part, sim%my_id
    n_part_total = n_part
    call MPI_ALLREDUCE(MPI_IN_PLACE, n_part_total, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD, ierr)
    n_limit = this%n_limit
    if (3*n_part_total .gt. n_limit*amount_processes) resample = .true.

    ! Resample when the tripling by the knock-on collisions would no longer fit
    if (resample) then
      if(sim%my_id .eq. 0) write(*,*) "Start resampling, necessary as n_part = ", n_part," and n_limit = ", n_limit
      start_time = MPI_WTIME()

      call do_binning_mpi(sim, this%group_num, n_part, n_limit, this%Nmombin, this%Ntor, &
                          this%element_part_s, this%element_part_t, this%new_markers_per_element, &
                          this%emptyind, iterations)

      end_time = MPI_WTIME()
      tot_time = mpi_minmeanmax(end_time-start_time)
      if (sim%my_id .eq. 0) then
          write(*,"(A,3f10.3,A)") , "Time taken for resampling finished in (min/mean/max): ", tot_time, " seconds"
      endif

      n_part = count(sim%groups(this%group_num)%particles(:)%i_elm .gt. 0)
      if (3*n_part .gt. n_limit) then
         if (sim%my_id .eq. 0) then 
          write(*,*) 'Settings of the resampling do not allow for a large enough decrease in amount of markers. Consider tuning the settings to have less volume bins or increase the size of the particle group.'
          write(*,*) 'After resampling and locally on id 0; n_part = ', n_part, ', n_limit = ', n_limit
         end if 
         stop
      end if
    end if

    if(this%each_nstep_part .eq. -9999991) then
      dt_coll = sim%nstep_inner_loop*sim%tstep_part_adj
    else
      dt_coll = this%each_nstep_part*sim%tstep_part_adj
    endif

    if (sim%my_id .eq. 0) write(*,*) 'Doing the knock-on collisions ...'
    start_time = MPI_WTIME()
    call Moller_scatt(sim, this%group_num, this%gamma_min, dt_coll, n_part, this%emptyind)
    call with(sim, counter)
    end_time = MPI_WTIME()
    tot_time = mpi_minmeanmax(end_time-start_time)
    if (sim%my_id .eq. 0) then
        write(*,"(A,3f10.3,A)") , "Time taken for knock-on collisions finished in (min/mean/max): ", tot_time, " seconds"
    endif

  end subroutine do_RE_avalanche

end module mod_re_avalanche

