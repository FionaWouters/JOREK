!> module containing initialization and pdf (probability distribution function) generation
!> subroutines specifically useful for runaway electrons
module initialisers_RE
  use mod_particle_types
  use mod_particle_sim
  use mod_rng
  use initialisers_base
  use constants, only: EL_CHG, ATOMIC_MASS_UNIT, SPEED_OF_LIGHT, MASS_ELECTRON, TWOPI
  use mod_pusher_tools, only: get_orthonormals
  use mod_coordinate_transforms, only: vector_cylindrical_to_cartesian
  use mod_model_settings
  use phys_module, only: CENTRAL_DENSITY, CENTRAL_MASS, ATOMIC_MASS_UNIT, MU_ZERO
  use equil_info
  implicit none

  contains

! Quick and rough function to sample markers based on RZ-coordinates
pure function RZ_pdf(var) result(p)
  real*8, intent(in)  :: var(2) ! var(1)=j
  real*8              :: p
  real*8              :: minor_r
  real*8              :: R_ax, Z_ax

  R_ax = ES%R_axis
  Z_ax = ES%Z_axis

  minor_r = sqrt((var(1)-Z_ax)**2 + (var(2)-R_ax)**2)

  p = 1 / (1 + minor_r)**2
end function RZ_pdf

pure function analytical_pdf(var) result(p)
  real*8, intent(in)  :: var(2) ! var(1)=j
  real*8              :: p
  real*8              :: minor_r
  real*8              :: nu
  real*8              :: R_ax, Z_ax
  real*8              :: LCFS_a

  R_ax   = ES%R_axis
  Z_ax   = ES%Z_axis
  LCFS_a = ES%LCFS_a
  nu     = 2.d0

  minor_r = sqrt((var(1)-Z_ax)**2 + (var(2)-R_ax)**2)

  p = (1.d0 - (minor_r/LCFS_a)**2)**nu

end function analytical_pdf

! Quick and rough function to sample markers proportionally to toroidal current density
pure function current_pdf(var) result(p)
  real*8, intent(in)  :: var(2) ! var(2)=j, var(1) = R
  !real*8, intent(in)  :: var(1) ! var(1)=j
  real*8              :: p
  real*8              :: jzmin, jzmax 

  !> temporarily hard coded, but should be able to be obtained from fluid restart file
  jzmax = 1.173 / 10
  jzmin = 0.0003166 / 11

  p = (abs(var(2))/var(1)-jzmin)/(jzmax-jzmin)

end function current_pdf

! Quick and rough function to sample markers proportionally to n_RE from fluid simulation
pure function density_pdf(var) result(p)
  real*8, intent(in)  :: var(2) ! var(2)=n_RE, var(1) = R
  !real*8, intent(in)  :: var(1) ! var(1)=j
  real*8              :: p
  real*8              :: nre_min, nre_max 

  !> temporarily hard coded, but should be able to be obtained from fluid restart file
  nre_max = 6.815d16 / 1.953d0 !1.173 / 10
  nre_min = -3.518d13 / 1.715d0 !0.0003166 / 11

  p = (var(2)/var(1)-nre_min)/(nre_max-nre_min)

end function density_pdf
    
subroutine basic_initialization(sim, group_num, rng, init_pdf, energy, pitch, std_energy)
  use phys_module, only: tstep_particles
  use mod_kinetic_relativistic
  use mod_sampling, only: boxmueller_transform

  type(particle_sim),                   intent(inout) :: sim
  integer,                              intent(in)    :: group_num
  class(type_rng),                      intent(in)    :: rng
  character(len=50),                    intent(in)    :: init_pdf
  real*8,                               intent(in)    :: energy, pitch ! Kinetic energy in units of eV and pitch
  real*8,               optional,       intent(in)    :: std_energy
  real*8,               allocatable                   :: p_tot(:), p_par(:), p_perp(:)
  integer                                             :: j
  real(kind=8)                                        :: psi, U, gyro_angle
  real(kind=8),         dimension(3)                  :: E, B, B_cart, B_norm
  real*8                                              :: e1(3), e2(3) 
  integer                                             :: num_part
  real*8,               allocatable                   :: ran_uniform(:), ran_gaussian(:)

  select case (trim(init_pdf))
    case ("RZ")
      call initialise_particles(sim%groups(group_num)%particles, sim%fields%node_list, sim%fields%element_list, rng, variables=[-2,-1], transform=RZ_pdf)
    case ("current")
      call initialise_particles(sim%groups(group_num)%particles, sim%fields%node_list, sim%fields%element_list, rng, variables=[-1,var_zj], transform=current_pdf)
    case ("density")
      call initialise_particles(sim%groups(group_num)%particles, sim%fields%node_list, sim%fields%element_list, rng, variables=[-1,8], transform=density_pdf)  
    case ("analytical")
      call initialise_particles(sim%groups(group_num)%particles, sim%fields%node_list, sim%fields%element_list, rng, variables=[-2,-1], transform=analytical_pdf)
    case default
      if (sim%my_id == 0) then
        write(*,*) "ERROR: ", trim(init_pdf), " is not a valid pdf/transform function for "
        write(*,*) "  for group '", sim%groups(group_num)%id, "' when using the 'basic_initialization' function" 
        endif
      stop 1
  end select

  num_part = size(sim%groups(group_num)%particles,1) 

  allocate(p_tot(num_part))
  allocate(p_par(num_part))
  allocate(p_perp(num_part))

  ! Generate gassian distributed energy
  if (present(std_energy)) then

    allocate(ran_uniform(num_part + mod(num_part,2)))
    allocate(ran_gaussian(num_part + mod(num_part,2)))

    call random_number(ran_uniform)
    ran_gaussian = boxmueller_transform(ran_uniform)

    p_tot               = sqrt(((energy + std_energy*ran_gaussian(:num_part))*EL_CHG/SPEED_OF_LIGHT + MASS_ELECTRON*SPEED_OF_LIGHT)**2 - (MASS_ELECTRON*SPEED_OF_LIGHT)**2)/ATOMIC_MASS_UNIT ! [AMU*m/s]

    deallocate(ran_uniform)
    deallocate(ran_gaussian)

  else

    p_tot               = sqrt((energy*EL_CHG/SPEED_OF_LIGHT + MASS_ELECTRON*SPEED_OF_LIGHT)**2 - (MASS_ELECTRON*SPEED_OF_LIGHT)**2)/ATOMIC_MASS_UNIT ! [AMU*m/s]

  end if

  p_par               = pitch * p_tot
  p_perp              = sqrt(p_tot**2 - p_par**2)

  ! Set particle momentum
  select type (particles => sim%groups(group_num)%particles)
  type is (particle_kinetic_relativistic)
    !$omp parallel do default(none) &
    !$omp private(E, B, psi, U, B_cart, B_norm, e1, e2, gyro_angle, j) &
    !$omp shared (sim, tstep_particles, p_par, p_perp, num_part)
    do j=1,num_part

      ! Extract magnetic field and convert to cartesian coordinates
      call sim%fields%calc_EBpsiU(sim%time, particles(j)%i_elm, particles(j)%st, particles(j)%x(3), E, B, psi, U)
      B_cart = vector_cylindrical_to_cartesian(particles(j)%x(3),B)

      B_norm = B_cart/norm2(B_cart)

      ! Generate perpendicular component based on sampled gyro angle
      call get_orthonormals(B_norm, e1, e2)
      call random_number(gyro_angle)
      gyro_angle = gyro_angle * TWOPI

      particles(j)%p = p_par(j) * B_norm + p_perp(j)*(e1*cos(gyro_angle) + e2*sin(gyro_angle))

    end do
    !$omp end parallel do 
  end select

  deallocate(p_tot)
  deallocate(p_par)
  deallocate(p_perp)

end subroutine basic_initialization

end module initialisers_RE