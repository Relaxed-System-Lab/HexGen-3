def roofline_analyze(bandwidth, max_OPS, OPs, memory_access):
    """
    Perform roofline performance model analysis to determine performance bottlenecks.

    The roofline model is a performance analysis tool that helps identify whether a
    computation is limited by memory bandwidth or compute capability. It plots
    performance (operations per second) against arithmetic intensity (operations per byte).

    Args:
        bandwidth (float): Memory bandwidth in bytes per second
        max_OPS (float): Maximum computational performance in operations per second (peak FLOPS)
        OPs (float): Total number of arithmetic operations performed
        memory_access (float): Total memory bytes accessed

    Returns:
        tuple: A 3-element tuple containing:
            - arithmetic_intensity (float): Operations per byte accessed (OPs/byte)
            - performance (float): Achieved performance in operations per second
            - bound (str): Performance limitation type - either "memory" or "compute"

    The function determines the performance bound by:
    1. Calculating the roofline turning point where memory and compute bounds intersect
    2. Computing the actual arithmetic intensity of the operation
    3. Determining if the operation is memory-bound or compute-bound
    4. Calculating the achievable performance based on the binding constraint
    """
  
    # Extract peak compute performance
    y_max = max_OPS
    memory_access_bytes = memory_access

    # Handle case when component is disabled (OPs = 0 and memory_access = 0)
    if OPs == 0 and memory_access_bytes == 0:
        return 0, 0, "disabled"  # Return zeros for disabled components

    # Calculate the roofline turning point: where memory and compute bounds intersect
    # This represents the minimum arithmetic intensity needed to be compute-bound
    turning_point = y_max / bandwidth

    # Calculate actual arithmetic intensity of the operation (operations per byte)
    if memory_access_bytes == 0:
        # If no memory access but OPs > 0, assume perfect arithmetic intensity (compute-bound)
        arithmetic_intensity = float('inf')
    else:
        arithmetic_intensity = OPs / memory_access_bytes

    # Determine performance limitation and calculate achievable performance
    if arithmetic_intensity < turning_point:
        # Memory-bound: performance limited by memory bandwidth
        bound = "memory"
        performance = arithmetic_intensity * bandwidth
    else:
        # Compute-bound: performance limited by peak compute capability
        bound = "compute"
        performance = y_max

    return arithmetic_intensity, performance, bound