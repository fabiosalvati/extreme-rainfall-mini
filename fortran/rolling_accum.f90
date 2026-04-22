program rolling_accumulation
    implicit none

    integer :: n
    integer :: window
    integer :: i
    integer :: j
    integer :: ios

    real(8) :: missing_value
    real(8) :: current_sum

    real(8), allocatable :: rain(:)
    real(8), allocatable :: rolling_sum(:)
    integer, allocatable :: valid_flag(:)

    character(len=200) :: input_file
    character(len=200) :: output_file

    logical :: bad_window

    ! Read command-line arguments:
    ! 1 = input file
    ! 2 = output file
    call get_command_argument(1, input_file)
    call get_command_argument(2, output_file)

    if (len_trim(input_file) == 0 .or. len_trim(output_file) == 0) then
        print *, "Usage: ./rolling_accumulation input.txt output.txt"
        stop
    end if

    ! Open input file
    open(unit=10, file=trim(input_file), status="old", action="read", iostat=ios)
    if (ios /= 0) then
        print *, "Could not open input file."
        stop
    end if

    ! Input format:
    ! line 1: n
    ! line 2: window
    ! line 3: missing_value
    ! then n rainfall values, one per line
    read(10, *) n
    read(10, *) window
    read(10, *) missing_value

    allocate(rain(n))
    allocate(rolling_sum(n))
    allocate(valid_flag(n))

    do i = 1, n
        read(10, *) rain(i)
    end do

    close(10)

    ! Initialize outputs
    do i = 1, n
        rolling_sum(i) = missing_value
        valid_flag(i) = 0
    end do

    ! Compute rolling sums
    do i = 1, n

        ! If there are not enough values yet, window is invalid
        if (i < window) then
            rolling_sum(i) = missing_value
            valid_flag(i) = 0

        else
            current_sum = 0.0d0
            bad_window = .false.

            do j = i - window + 1, i
                if (rain(j) == missing_value) then
                    bad_window = .true.
                else
                    current_sum = current_sum + rain(j)
                end if
            end do

            if (bad_window) then
                rolling_sum(i) = missing_value
                valid_flag(i) = 0
            else
                rolling_sum(i) = current_sum
                valid_flag(i) = 1
            end if
        end if
    end do

    ! Write output file
    open(unit=20, file=trim(output_file), status="replace", action="write", iostat=ios)
    if (ios /= 0) then
        print *, "Could not open output file."
        stop
    end if

    write(20, *) "# index rainfall rolling_sum valid_flag"

    do i = 1, n
        write(20, "(I4,1X,F10.3,1X,F10.3,1X,I1)") i, rain(i), rolling_sum(i), valid_flag(i)
    end do

    close(20)

    print *, "Done."
    print *, "Input file:  ", trim(input_file)
    print *, "Output file: ", trim(output_file)

    deallocate(rain)
    deallocate(rolling_sum)
    deallocate(valid_flag)

end program rolling_accumulation