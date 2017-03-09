#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cassert>
#include <chrono>

#include <random>

#ifdef __APPLE__
#include "OpenCL/opencl.h"
#else
#include "CL/cl.h"
#endif

//#include "omp.h"

std::string GenerateRandomNucleotideString(size_t length) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 3);

    std::vector<char> vec(length);

    std::generate_n(vec.begin(), length, [&dis, &gen]() {
        switch(dis(gen)) {
            case 0:
                return 'A';
            case 1:
                return 'C';
            case 2:
                return 'G';
            case 3:
                return 'T';
            default:
                throw std::logic_error("Random number generator too large");
        }
    });

    return std::string(vec.begin(), vec.end());
}

template <class T>
class Matrix {
public:
    Matrix(size_t num_rows, size_t num_cols, const T & value) : num_rows_(num_rows), num_cols_(num_cols) {
        mat_ = new T*[num_rows];
        for (size_t r = 0; r < num_rows; ++r) {
            mat_[r] = new T[num_cols];
            for (size_t c = 0; c < num_cols; ++c) {
                mat_[r][c] = value;
            }
        }
    }

    ~Matrix() {
        for (size_t r = 0; r < num_rows_; ++r) {
            delete [] mat_[r];
        }
        delete [] mat_;
    }

    T* operator[] (size_t row_index) { return mat_[row_index]; }

    size_t GetNumRows() { return num_rows_; }
    size_t GetNumCols() { return num_cols_; }

private:
    T** mat_;

    size_t num_rows_;
    size_t num_cols_;
    
    std::vector<T> vec_;
};

template <class T>
class RowBuffer {
public:
    RowBuffer(size_t length, const T & value) : length_(length) {
        arr_ = new T[length];
        for (size_t k = 0; k < length; ++k) {
            arr_[k] = value;
        }
    }

    ~RowBuffer() {
        delete [] arr_;
    }

    RowBuffer(const RowBuffer & other) = delete;
//    RowBuffer(RowBuffer && other) {
//        arr_ = other.arr_;
//        length_ = other.length_;
//        other.arr_ = nullptr;
//    }

    RowBuffer(RowBuffer && other) = delete;

    RowBuffer& operator=(const RowBuffer & other) = delete;
    RowBuffer& operator=(RowBuffer && other) {
        assert(other.length_ == length_);

        T* temp = other.arr_;
        other.arr_ = arr_;
        arr_ = temp;

        return *this;
    }

    T& operator[] (size_t index) { return arr_[index]; }

    size_t GetLength() { return length_; }
private:
    T* arr_;

    size_t length_;
};

const char *getErrorString(cl_int error)
{
    switch(error){
            // run-time and JIT compiler errors
        case 0: return "CL_SUCCESS";
        case -1: return "CL_DEVICE_NOT_FOUND";
        case -2: return "CL_DEVICE_NOT_AVAILABLE";
        case -3: return "CL_COMPILER_NOT_AVAILABLE";
        case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case -5: return "CL_OUT_OF_RESOURCES";
        case -6: return "CL_OUT_OF_HOST_MEMORY";
        case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case -8: return "CL_MEM_COPY_OVERLAP";
        case -9: return "CL_IMAGE_FORMAT_MISMATCH";
        case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case -11: return "CL_BUILD_PROGRAM_FAILURE";
        case -12: return "CL_MAP_FAILURE";
        case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case -15: return "CL_COMPILE_PROGRAM_FAILURE";
        case -16: return "CL_LINKER_NOT_AVAILABLE";
        case -17: return "CL_LINK_PROGRAM_FAILURE";
        case -18: return "CL_DEVICE_PARTITION_FAILED";
        case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

            // compile-time errors
        case -30: return "CL_INVALID_VALUE";
        case -31: return "CL_INVALID_DEVICE_TYPE";
        case -32: return "CL_INVALID_PLATFORM";
        case -33: return "CL_INVALID_DEVICE";
        case -34: return "CL_INVALID_CONTEXT";
        case -35: return "CL_INVALID_QUEUE_PROPERTIES";
        case -36: return "CL_INVALID_COMMAND_QUEUE";
        case -37: return "CL_INVALID_HOST_PTR";
        case -38: return "CL_INVALID_MEM_OBJECT";
        case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case -40: return "CL_INVALID_IMAGE_SIZE";
        case -41: return "CL_INVALID_SAMPLER";
        case -42: return "CL_INVALID_BINARY";
        case -43: return "CL_INVALID_BUILD_OPTIONS";
        case -44: return "CL_INVALID_PROGRAM";
        case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case -46: return "CL_INVALID_KERNEL_NAME";
        case -47: return "CL_INVALID_KERNEL_DEFINITION";
        case -48: return "CL_INVALID_KERNEL";
        case -49: return "CL_INVALID_ARG_INDEX";
        case -50: return "CL_INVALID_ARG_VALUE";
        case -51: return "CL_INVALID_ARG_SIZE";
        case -52: return "CL_INVALID_KERNEL_ARGS";
        case -53: return "CL_INVALID_WORK_DIMENSION";
        case -54: return "CL_INVALID_WORK_GROUP_SIZE";
        case -55: return "CL_INVALID_WORK_ITEM_SIZE";
        case -56: return "CL_INVALID_GLOBAL_OFFSET";
        case -57: return "CL_INVALID_EVENT_WAIT_LIST";
        case -58: return "CL_INVALID_EVENT";
        case -59: return "CL_INVALID_OPERATION";
        case -60: return "CL_INVALID_GL_OBJECT";
        case -61: return "CL_INVALID_BUFFER_SIZE";
        case -62: return "CL_INVALID_MIP_LEVEL";
        case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
        case -64: return "CL_INVALID_PROPERTY";
        case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
        case -66: return "CL_INVALID_COMPILER_OPTIONS";
        case -67: return "CL_INVALID_LINKER_OPTIONS";
        case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

            // extension errors
        case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
        case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
        case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
        case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
        case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
        case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
        default: return "Unknown OpenCL error";
    }
}

void CheckError (cl_int error)
{
    if (error != CL_SUCCESS) {
        std::cerr << "OpenCL call failed with error " << error << std::endl;
        std::cerr << getErrorString(error) << std::endl;
        throw std::runtime_error(getErrorString(error));
    }
}


std::vector<char> ReadKernelFromFilename(const std::string & filename) {
    std::ifstream input_file(filename, std::ios_base::in | std::ios_base::binary);
    input_file.seekg(0, std::ios::end);
    auto length = input_file.tellg();
    std::vector<char> vec(length);
    input_file.seekg(0, std::ios::beg);
    input_file.read(vec.data(), length);
    return vec;
}

std::string GetPlatformName (cl_platform_id id)
{
    size_t size = 0;
    clGetPlatformInfo (id, CL_PLATFORM_NAME, 0, nullptr, &size);

    std::string result;
    result.resize (size);
    clGetPlatformInfo (id, CL_PLATFORM_NAME, size,
                       const_cast<char*> (result.data ()), nullptr);

    return result;
}

std::string GetDeviceName (cl_device_id id)
{
    size_t size = 0;
    clGetDeviceInfo (id, CL_DEVICE_NAME, 0, nullptr, &size);

    std::string result;
    result.resize (size);
    clGetDeviceInfo (id, CL_DEVICE_NAME, size,
                     const_cast<char*> (result.data ()), nullptr);

    return result;
}

namespace cl {
    struct DeviceInfo {
        cl_uint device_address_bits;
        cl_bool device_available;
        cl_bool device_compiler_available;
        cl_device_fp_config device_double_fp_config;
        cl_bool device_endian_little;
        cl_bool device_error_correction_support;
        cl_device_exec_capabilities device_execution_capabilities;
        std::string device_extensions;
        cl_ulong device_global_mem_cache_size;
        cl_device_mem_cache_type device_global_mem_cache_type;
        cl_uint device_global_mem_cacheline_size;
        cl_ulong device_global_mem_size;
        cl_device_fp_config device_half_fp_config;
        cl_bool device_image_support;
        size_t device_image2d_max_height;
        size_t device_image2d_max_width;
        size_t device_image3d_max_depth;
        size_t device_image3d_max_height;
        size_t device_image3d_max_width;
        cl_ulong device_local_mem_size;
        cl_device_local_mem_type device_local_mem_type;
        cl_uint device_max_clock_frequency;
        cl_uint device_max_compute_units;
        cl_uint device_max_constant_args;
        cl_ulong device_max_constant_buffer_size;
        cl_ulong device_max_mem_alloc_size;
        size_t device_max_parameter_size;
        cl_uint device_max_read_image_args;
        cl_uint device_max_samplers;
        size_t device_max_work_group_size;
        cl_uint device_max_work_item_dimensions;
        std::vector<size_t> device_max_work_item_sizes;
        cl_uint device_max_write_image_args;
        cl_uint device_mem_base_addr_align;
        cl_uint device_min_data_type_align_size;
        std::string device_name;
        cl_platform_id device_platform;
        cl_uint device_preferred_vector_width_char;
        cl_uint device_preferred_vector_width_short;
        cl_uint device_preferred_vector_width_int;
        cl_uint device_preferred_vector_width_long;
        cl_uint device_preferred_vector_width_float;
        cl_uint device_preferred_vector_width_double;
        std::string device_profile;
        size_t device_profiling_timer_resolution; // in nanoseconds
        cl_command_queue_properties device_queue_properties;
        cl_device_fp_config device_single_fp_config;
        cl_device_type device_type;
        std::string device_vendor;
        cl_uint device_vendor_id;
        std::string device_version;
        std::string driver_version;
    };
}

cl::DeviceInfo GetDeviceInfo(cl_device_id device_id) {
    cl::DeviceInfo device_info;

    #define GETDEVICEINFO(param_name, struct_name) clGetDeviceInfo(device_id, param_name, sizeof(device_info.struct_name), &device_info.struct_name, nullptr);

    cl_int error;
    error = clGetDeviceInfo(device_id, CL_DEVICE_ADDRESS_BITS, sizeof(device_info.device_address_bits), &device_info.device_address_bits, nullptr);
    CheckError(error);

    error = GETDEVICEINFO(CL_DEVICE_AVAILABLE, device_available);
    CheckError(error);

    error = GETDEVICEINFO(CL_DEVICE_COMPILER_AVAILABLE, device_compiler_available);
    CheckError(error);

    error = GETDEVICEINFO(CL_DEVICE_DOUBLE_FP_CONFIG, device_double_fp_config);
    CheckError(error);

    error = GETDEVICEINFO(CL_DEVICE_ENDIAN_LITTLE, device_endian_little);
    CheckError(error);

    error = GETDEVICEINFO(CL_DEVICE_ERROR_CORRECTION_SUPPORT, device_error_correction_support);
    CheckError(error);

    error = GETDEVICEINFO(CL_DEVICE_EXECUTION_CAPABILITIES, device_execution_capabilities);
    CheckError(error);

    size_t device_extensions_str_size;
    error = clGetDeviceInfo(device_id, CL_DEVICE_EXTENSIONS, 0, nullptr, &device_extensions_str_size);
    CheckError(error);
    std::vector<char> device_extensions_vec(device_extensions_str_size);
    error = clGetDeviceInfo(device_id, CL_DEVICE_EXTENSIONS, device_extensions_str_size, device_extensions_vec.data(), nullptr);
    CheckError(error);
    device_info.device_extensions = std::string(device_extensions_vec.begin(), device_extensions_vec.end());
    CheckError(error);

    error = GETDEVICEINFO(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, device_global_mem_cache_size);
    CheckError(error);

    error = GETDEVICEINFO(CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, device_global_mem_cache_type);
    CheckError(error);

    error = GETDEVICEINFO(CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, device_global_mem_cacheline_size);
    CheckError(error);

    error = GETDEVICEINFO(CL_DEVICE_GLOBAL_MEM_SIZE, device_global_mem_size);
    CheckError(error);

//    error = GETDEVICEINFO(CL_DEVICE_HALF_FP_CONFIG, device_half_fp_config);
//    CheckError(error);
// Possibly bugged in OSX?

    error = GETDEVICEINFO(CL_DEVICE_IMAGE_SUPPORT, device_image_support);
    CheckError(error);

    error = GETDEVICEINFO(CL_DEVICE_IMAGE2D_MAX_HEIGHT, device_image2d_max_height);
    CheckError(error);

    error = GETDEVICEINFO(CL_DEVICE_IMAGE2D_MAX_WIDTH, device_image2d_max_width);
    CheckError(error);

    error = GETDEVICEINFO(CL_DEVICE_IMAGE3D_MAX_DEPTH, device_image3d_max_depth);
    CheckError(error);

    error = GETDEVICEINFO(CL_DEVICE_IMAGE3D_MAX_HEIGHT, device_image3d_max_height);
    CheckError(error);

    error = GETDEVICEINFO(CL_DEVICE_IMAGE3D_MAX_WIDTH, device_image3d_max_width);
    CheckError(error);

    error = GETDEVICEINFO(CL_DEVICE_LOCAL_MEM_SIZE, device_local_mem_size);
    CheckError(error);

    error = GETDEVICEINFO(CL_DEVICE_LOCAL_MEM_TYPE, device_local_mem_type);
    CheckError(error);

    error = GETDEVICEINFO(CL_DEVICE_MAX_CLOCK_FREQUENCY, device_max_clock_frequency);
    CheckError(error);

    error = GETDEVICEINFO(CL_DEVICE_MAX_COMPUTE_UNITS, device_max_compute_units);
    CheckError(error);

    error = GETDEVICEINFO(CL_DEVICE_MAX_CONSTANT_ARGS, device_max_constant_args);
    CheckError(error);

    error = GETDEVICEINFO(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, device_max_constant_buffer_size);
    CheckError(error);

    error = GETDEVICEINFO(CL_DEVICE_MAX_MEM_ALLOC_SIZE, device_max_mem_alloc_size);
    CheckError(error);

    error = GETDEVICEINFO(CL_DEVICE_MAX_PARAMETER_SIZE, device_max_parameter_size);
    CheckError(error);

    error = GETDEVICEINFO(CL_DEVICE_MAX_READ_IMAGE_ARGS, device_max_read_image_args);
    CheckError(error);

    error = GETDEVICEINFO(CL_DEVICE_MAX_SAMPLERS, device_max_samplers);
    CheckError(error);

    error = GETDEVICEINFO(CL_DEVICE_MAX_WORK_GROUP_SIZE, device_max_work_group_size);
    CheckError(error);

    error = GETDEVICEINFO(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, device_max_work_item_dimensions);
    CheckError(error);

//    error = GETDEVICEINFO(CL_DEVICE_MAX_WORK_ITEM_SIZES, device_max_work_item_sizes); // vector
//    CheckError(error);

    error = GETDEVICEINFO(CL_DEVICE_MAX_WRITE_IMAGE_ARGS, device_max_write_image_args);
    CheckError(error);

    error = GETDEVICEINFO(CL_DEVICE_MEM_BASE_ADDR_ALIGN, device_mem_base_addr_align);
    CheckError(error);

    error = GETDEVICEINFO(CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, device_min_data_type_align_size);
    CheckError(error);

    size_t device_name_str_size;
    error = clGetDeviceInfo(device_id, CL_DEVICE_NAME, 0, nullptr, &device_name_str_size);
    CheckError(error);
    std::vector<char> device_name_vec(device_name_str_size);
    error = clGetDeviceInfo(device_id, CL_DEVICE_NAME, device_name_str_size, device_name_vec.data(), nullptr);
    CheckError(error);
    device_info.device_name = std::string(device_name_vec.begin(), device_name_vec.end());
    CheckError(error);


    error = GETDEVICEINFO(CL_DEVICE_PLATFORM, device_platform);
    CheckError(error);

    error = GETDEVICEINFO(CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, device_preferred_vector_width_char);
    CheckError(error);

    error = GETDEVICEINFO(CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, device_preferred_vector_width_short);
    CheckError(error);

    error = GETDEVICEINFO(CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, device_preferred_vector_width_int);
    CheckError(error);

    error = GETDEVICEINFO(CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, device_preferred_vector_width_long);
    CheckError(error);

    error = GETDEVICEINFO(CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, device_preferred_vector_width_float);
    CheckError(error);

    error = GETDEVICEINFO(CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, device_preferred_vector_width_double);
    CheckError(error);

    size_t device_profile_str_size;
    error = clGetDeviceInfo(device_id, CL_DEVICE_PROFILE, 0, nullptr, &device_profile_str_size);
    CheckError(error);
    std::vector<char> device_profile_vec(device_profile_str_size);
    error = clGetDeviceInfo(device_id, CL_DEVICE_PROFILE, device_profile_str_size, device_profile_vec.data(), nullptr);
    CheckError(error);
    device_info.device_profile = std::string(device_profile_vec.begin(), device_profile_vec.end());
    CheckError(error);

    error = GETDEVICEINFO(CL_DEVICE_PROFILING_TIMER_RESOLUTION, device_profiling_timer_resolution);
    CheckError(error);

    error = GETDEVICEINFO(CL_DEVICE_QUEUE_PROPERTIES, device_queue_properties);
    CheckError(error);

    error = GETDEVICEINFO(CL_DEVICE_SINGLE_FP_CONFIG, device_single_fp_config);
    CheckError(error);

    error = GETDEVICEINFO(CL_DEVICE_TYPE, device_type);
    CheckError(error);

    size_t device_vendor_str_size;
    error = clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, 0, nullptr, &device_vendor_str_size);
    CheckError(error);
    std::vector<char> device_vendor_vec(device_vendor_str_size);
    error = clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, device_vendor_str_size, device_vendor_vec.data(), nullptr);
    CheckError(error);
    device_info.device_vendor = std::string(device_vendor_vec.begin(), device_vendor_vec.end());
    CheckError(error);

    error = GETDEVICEINFO(CL_DEVICE_VENDOR_ID, device_vendor_id);
    CheckError(error);

    size_t device_version_str_size;
    error = clGetDeviceInfo(device_id, CL_DEVICE_VERSION, 0, nullptr, &device_version_str_size);
    CheckError(error);
    std::vector<char> device_version_vec(device_version_str_size);
    error = clGetDeviceInfo(device_id, CL_DEVICE_VERSION, device_version_str_size, device_version_vec.data(), nullptr);
    CheckError(error);
    device_info.device_version = std::string(device_version_vec.begin(), device_version_vec.end());
    CheckError(error);

    size_t driver_version_str_size;
    error = clGetDeviceInfo(device_id, CL_DRIVER_VERSION, 0, nullptr, &driver_version_str_size);
    CheckError(error);
    std::vector<char> driver_version_vec(driver_version_str_size);
    error = clGetDeviceInfo(device_id, CL_DRIVER_VERSION, driver_version_str_size, driver_version_vec.data(), nullptr);
    CheckError(error);
    device_info.driver_version = std::string(driver_version_vec.begin(), driver_version_vec.end());
    CheckError(error);

    return device_info;

#undef GETDEVICEINFO
}

void PrintDeviceInfo(cl_device_id device_id) {
    auto info = GetDeviceInfo(device_id);

    std::cout << "Address Bits: " << info.device_address_bits << std::endl;
    std::cout << "Available: " << info.device_available << std::endl;
    std::cout << "Compiler Available: " << info.device_compiler_available << std::endl;
    std::cout << "Double FP Config: " << info.device_double_fp_config << std::endl;
    std::cout << "Little Endian: " << info.device_endian_little << std::endl;
    std::cout << "ECC Support: " << info.device_error_correction_support << std::endl;
    std::cout << "Execution Capabilities: " << info.device_execution_capabilities << std::endl;
    std::cout << "Device Extensions: " << info.device_extensions << std::endl;
    std::cout << "Global Mem Cache Size: " << info.device_global_mem_cache_size << std::endl;
    std::cout << "Global Mem Cache Type: " << info.device_global_mem_cache_type << std::endl;
    std::cout << "Global Mem Cacheline Size: " << info.device_global_mem_cacheline_size << std::endl;
    std::cout << "Global Mem Size: " << info.device_global_mem_size << std::endl;
//    std::cout << "Half FP Config: " << info.device_half_fp_config << std::endl;
    std::cout << "Image support: " << info.device_image_support << std::endl;
    std::cout << "2D Image Max Height: " << info.device_image2d_max_height << std::endl;
    std::cout << "2D Image Max Width: " << info.device_image2d_max_width << std::endl;
    std::cout << "3D Image Max Depth: " << info.device_image3d_max_depth << std::endl;
    std::cout << "3D Image Max Height: " << info.device_image3d_max_height << std::endl;
    std::cout << "3D Image Max Width: " << info.device_image3d_max_width << std::endl;
    std::cout << "Local Mem Size: " << info.device_local_mem_size << std::endl;
    std::cout << "Local Mem Type: " << info.device_local_mem_type << std::endl;
    std::cout << "Max Clock Freq (MHz): " << info.device_max_clock_frequency << std::endl;
    std::cout << "Max Compute Units: " << info.device_max_compute_units << std::endl;
    std::cout << "Max Const Args: " << info.device_max_constant_args << std::endl;
    std::cout << "Max Const Buffer Size: " << info.device_max_constant_buffer_size << std::endl;
    std::cout << "Max Allocation Size: " << info.device_max_mem_alloc_size << std::endl;
    std::cout << "Max Parameter Size: " << info.device_max_parameter_size << std::endl;
    std::cout << "Max Read Image Args: " << info.device_max_read_image_args << std::endl;
    std::cout << "Max Samplers: " << info.device_max_samplers << std::endl;
    std::cout << "Max Work Group Size: " << info.device_max_work_group_size << std::endl;
    std::cout << "Max Work Item Dimensions: " << info.device_max_work_item_dimensions << std::endl;
//    std::cout << "Max work item sizes: " << info.device_max_work_item_sizes << std::endl; //vector
    std::cout << "Max Write Image Args: " << info.device_max_write_image_args << std::endl;
    std::cout << "Mem Base Addr Alignment: " << info.device_mem_base_addr_align << std::endl;
    std::cout << "Min Data Type Alignment: " << info.device_min_data_type_align_size << std::endl;
    std::cout << "Device Name: " << info.device_name << std::endl;
    std::cout << "Device Platform: " << info.device_platform << std::endl;
    std::cout << "Preferred Vector Width (Char): " << info.device_preferred_vector_width_char << std::endl;
    std::cout << "Preferred Vector Width (Short): " << info.device_preferred_vector_width_short << std::endl;
    std::cout << "Preferred Vector Width (Int): " << info.device_preferred_vector_width_int << std::endl;
    std::cout << "Preferred Vector Width (Long): " << info.device_preferred_vector_width_long << std::endl;
    std::cout << "Preferred Vector Width (Float): " << info.device_preferred_vector_width_float << std::endl;
    std::cout << "Preferred Vector Width (Double): " << info.device_preferred_vector_width_double << std::endl;
    std::cout << "Profile: " << info.device_profile << std::endl;
    std::cout << "Timer Resolution (ns): " << info.device_profiling_timer_resolution << std::endl;
    std::cout << "Queue Properties: " << info.device_queue_properties << std::endl;
    std::cout << "Single FP Config: " << info.device_single_fp_config << std::endl;
    std::cout << "Device Type: " << info.device_type << std::endl;
    std::cout << "Device Vendor: " << info.device_vendor << std::endl;
    std::cout << "Device Vendor ID: " << info.device_vendor_id << std::endl;
    std::cout << "Device Version: " << info.device_version << std::endl;
    std::cout << "Driver Version: " << info.driver_version << std::endl;
}

size_t GetPaddedRowSize(size_t input_row_size) {
    --input_row_size;
    input_row_size |= input_row_size >> 1;
    input_row_size |= input_row_size >> 2;
    input_row_size |= input_row_size >> 4;
    input_row_size |= input_row_size >> 8;
    input_row_size |= input_row_size >> 16;
    input_row_size |= input_row_size >> 32;
    ++input_row_size;
    // std::cout << "Row size rounded up to next power of 2: " << padded_row_size << std::endl;

    return input_row_size;
}

int main ()
{
    cl_uint platformIdCount = 0;
    clGetPlatformIDs (0, nullptr, &platformIdCount);

    if (platformIdCount == 0) {
        std::cerr << "No OpenCL platform found" << std::endl;
        return 1;
    } else {
        std::cout << "Found " << platformIdCount << " platform(s)" << std::endl;
    }

    std::vector<cl_platform_id> platformIds (platformIdCount);
    clGetPlatformIDs (platformIdCount, platformIds.data(), nullptr);

    for (cl_uint i = 0; i < platformIdCount; ++i) {
        std::cout << "\t (" << (i+1) << ") : " << GetPlatformName (platformIds [i]) << std::endl;
    }

    cl_uint deviceIdCount = 0;
    clGetDeviceIDs (platformIds[0], CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceIdCount);

    if (deviceIdCount == 0) {
        std::cerr << "No OpenCL devices found" << std::endl;
        return 1;
    } else {
        std::cout << "Found " << deviceIdCount << " device(s)" << std::endl;
    }

    std::vector<cl_device_id> deviceIds (deviceIdCount);
    clGetDeviceIDs (platformIds [0], CL_DEVICE_TYPE_ALL, deviceIdCount,
                    deviceIds.data (), nullptr);

    for (cl_uint i = 0; i < deviceIdCount; ++i) {
        std::cout << "\t (" << (i+1) << ") : " << GetDeviceName (deviceIds [i]) << std::endl;
    }

    const cl_context_properties contextProperties [] = { CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platformIds[0]), 0, 0 };

    cl_int error = CL_SUCCESS;
    cl_context context = clCreateContext (contextProperties, deviceIdCount, deviceIds.data (), nullptr, nullptr, &error);
    CheckError (error);
    
    std::cout << "Context created" << std::endl;

    size_t DEVICE_NUMBER=0;

    PrintDeviceInfo(deviceIds[DEVICE_NUMBER]);

    cl_command_queue command_queue = clCreateCommandQueue (context, deviceIds[DEVICE_NUMBER], 0, &error);
    CheckError (error);

    cl_uint count = 1;

    // Here we're ready to actually run the code
    std::vector<char> kernel_bytes = ReadKernelFromFilename("C:/SmithWatermanOpenCL/src/SW_kernels.cl");

    std::string kernel_bytes_string(kernel_bytes.begin(), kernel_bytes.end());

    const char * source = kernel_bytes_string.c_str();
    size_t sourceSize[] = {strlen(source)};

    cl_program program = clCreateProgramWithSource(context, count, &source, sourceSize, &error);
    CheckError(error);

    clFinish(command_queue);

    error = clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);
    if (error != CL_SUCCESS) {
        std::cerr << "OpenCL call failed with error " << error << std::endl;
        std::cerr << getErrorString(error) << std::endl;
        size_t build_log_size;
        cl_int build_info_error = clGetProgramBuildInfo(program, deviceIds[DEVICE_NUMBER], CL_PROGRAM_BUILD_LOG, 0, nullptr, &build_log_size);
        CheckError(build_info_error);
        std::vector<char> error_buffer_vec(build_log_size);
        build_info_error = clGetProgramBuildInfo(program, deviceIds[DEVICE_NUMBER], CL_PROGRAM_BUILD_LOG, error_buffer_vec.size(), error_buffer_vec.data(), nullptr);
        CheckError(build_info_error);
        std::cerr << error_buffer_vec.data() << std::endl;
        throw std::runtime_error(getErrorString(error));
    }

    cl_kernel calc_fmat_row_kernel = clCreateKernel(program, "calc_fmat_row", &error);
    CheckError(error);

    cl_kernel upsweep_kernel = clCreateKernel(program, "upsweep", &error);
    CheckError(error);

    cl_kernel downsweep_kernel = clCreateKernel(program, "downsweep", &error);
    CheckError(error);

    using DataType = int32_t;

    DataType match = 5;
    DataType mismatch = -3;
    DataType gap_start_penalty = -8;
    DataType gap_extend_penalty = -1;

    //std::string seq1 = "CAGCCTCGCTTAG";
    //std::string seq2 = "AATGCCATTGCCGG";

    std::string seq1 = GenerateRandomNucleotideString(20'000'000); // columns
    std::string seq2 = GenerateRandomNucleotideString(150); // rows

    std::cout << "seq1.size(): " << seq1.size() << std::endl;
    std::cout << "seq2.size(): " << seq2.size() << std::endl;

    //Matrix<DataType> e_mat(seq2.size() + 1, seq1.size() + 1, 0);
    //Matrix<DataType> f_mat(seq2.size() + 1, seq1.size() + 1, 0);
    Matrix<DataType> h_mat(seq2.size() + 1, seq1.size() + 1, 0);
    //Matrix<DataType> h_hat_mat(seq2.size() + 1, seq1.size() + 1, 0);

    RowBuffer<DataType> e_mat_row_buffer(seq1.size() + 1, 0);
    RowBuffer<DataType> f_mat_row_buffer(seq1.size() + 1, 0);
    RowBuffer<DataType> f_mat_prev_row_buffer(seq1.size() + 1, 0);
    RowBuffer<DataType> h_hat_mat_row_buffer(seq1.size() + 1, 0);

    const size_t padded_row_size = GetPaddedRowSize(e_mat_row_buffer.GetLength());

    std::cout << "Padded row size: " << padded_row_size << std::endl;

    auto pow_of_2 = [](const size_t pow)
    {
        return 1 << pow;
    };


    auto log2 = [](size_t num) {
        size_t log = 0;

        while (num != 0) {
            num = num >> 1;
            ++log;
        }

        return log-1;
    };

    cl_mem padded_row_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(DataType) * padded_row_size, NULL, &error);
    CheckError(error);

    auto start = std::chrono::steady_clock::now();
    for (size_t r = 1; r < h_mat.GetNumRows(); ++r) {
        // Calculate f_mat_row on host
#pragma omp parallel for
        for (int64_t c = 1; c < e_mat_row_buffer.GetLength(); ++c) {
            f_mat_row_buffer[c] = std::max(f_mat_prev_row_buffer[c], h_mat[r-1][c] + gap_start_penalty) + gap_extend_penalty;
        }
#pragma omp parallel for
        for (int64_t c = 1; c < e_mat_row_buffer.GetLength(); ++c) {
            h_hat_mat_row_buffer[c] = std::max(std::max(h_mat[r-1][c-1] + (seq2.at(r-1) == seq1.at(c-1) ? match : mismatch), f_mat_row_buffer[c]), static_cast<DataType>(0));
        }

        {
            // initialize padded_row
            DataType * padded_row_host_ptr = (DataType *)clEnqueueMapBuffer(command_queue, padded_row_buffer, CL_TRUE, CL_MAP_WRITE, 0, sizeof(DataType) * padded_row_size, 0, NULL, NULL, &error);
			CheckError(error);
#pragma omp parallel for
            for (int64_t c = 0; c < h_hat_mat_row_buffer.GetLength(); ++c) {
                padded_row_host_ptr[c] = h_hat_mat_row_buffer[c];
            }
#pragma omp parallel for
            for (int64_t c = h_hat_mat_row_buffer.GetLength(); c < padded_row_size; ++c) {
                padded_row_host_ptr[c] = 0;
            }

            error = clEnqueueUnmapMemObject(command_queue, padded_row_buffer, padded_row_host_ptr, 0, nullptr, nullptr);
            CheckError(error);
        }

        // Upsweep
        for (size_t depth = 0; depth < log2(padded_row_size); ++depth) {
            error = 0;
            error = clSetKernelArg(upsweep_kernel, 0, sizeof(cl_mem), &padded_row_buffer);
            error |= clSetKernelArg(upsweep_kernel, 1, sizeof(cl_int), &depth);
            CheckError(error);

            size_t global = padded_row_size / pow_of_2(depth+1);
            error = clEnqueueNDRangeKernel(command_queue, upsweep_kernel, 1, NULL, &global, nullptr, 0, nullptr, nullptr);
            CheckError(error);
        }

        {
            DataType * padded_row_host_ptr = (DataType *)clEnqueueMapBuffer(command_queue, padded_row_buffer, CL_TRUE, CL_MAP_WRITE, 0, sizeof(DataType) * padded_row_size, 0, NULL, NULL, &error);
			CheckError(error);

            padded_row_host_ptr[padded_row_size-1] = 0;

            clEnqueueUnmapMemObject(command_queue, padded_row_buffer, padded_row_host_ptr, 0, nullptr, nullptr);
        }

        // Downsweep
        for (int64_t depth = log2(padded_row_size) - 1; depth >= 0; --depth) {
            error = 0;
            error = clSetKernelArg(downsweep_kernel, 0, sizeof(cl_mem), &padded_row_buffer);
            error |= clSetKernelArg(downsweep_kernel, 1, sizeof(cl_int), &depth);
            CheckError(error);

            size_t global = padded_row_size / pow_of_2(depth+1);
            error = clEnqueueNDRangeKernel(command_queue, downsweep_kernel, 1, NULL, &global, nullptr, 0, nullptr, nullptr);
            CheckError(error);
        }

        {
            DataType * padded_row_host_ptr = (DataType *)clEnqueueMapBuffer(command_queue, padded_row_buffer, CL_TRUE, CL_MAP_READ, 0, sizeof(DataType) * padded_row_size, 0, NULL, NULL, &error);
			CheckError(error);
#pragma omp parallel for
            for (int64_t c = 0; c < e_mat_row_buffer.GetLength(); ++c) {
                e_mat_row_buffer[c] = padded_row_host_ptr[c];
            }

            clEnqueueUnmapMemObject(command_queue, padded_row_buffer, padded_row_host_ptr, 0, nullptr, nullptr);
        }
#pragma omp parallel for
        for (int64_t c = 0; c < e_mat_row_buffer.GetLength(); ++c) {
            h_mat[r][c] = std::max(h_hat_mat_row_buffer[c], e_mat_row_buffer[c] + gap_start_penalty);
        }

        f_mat_prev_row_buffer = std::move(f_mat_row_buffer);
    }
    auto stop = std::chrono::steady_clock::now();

    std::cout << "SW took: " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << " ms" << std::endl;

//    for (int r = 0; r < h_mat.GetNumRows(); ++r) {
//        for (int c = 0; c < h_mat.GetNumCols(); ++c) {
//            std::cout << h_mat[r][c] << "\t";
//        }
//        std::cout << "\n";
//    }
//    std::cout << std::endl;

    clReleaseMemObject(padded_row_buffer);
    clReleaseProgram(program);
    clReleaseKernel(calc_fmat_row_kernel);
    clReleaseKernel(upsweep_kernel);
    clReleaseKernel(downsweep_kernel);
    clReleaseCommandQueue(command_queue);
    clReleaseContext (context);
}
