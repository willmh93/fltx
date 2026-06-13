if(NOT DEFINED FLTX_F256_EXPR_BENCH_BINARY_DIR)
    message(FATAL_ERROR "FLTX_F256_EXPR_BENCH_BINARY_DIR is required.")
endif()

if(NOT DEFINED FLTX_F256_EXPR_BENCH_INCLUDE_DIR)
    message(FATAL_ERROR "FLTX_F256_EXPR_BENCH_INCLUDE_DIR is required.")
endif()

if(NOT DEFINED FLTX_F256_EXPR_BENCH_COMPILER)
    message(FATAL_ERROR "FLTX_F256_EXPR_BENCH_COMPILER is required.")
endif()

if(NOT DEFINED FLTX_F256_EXPR_BENCH_COMPILER_ID)
    set(FLTX_F256_EXPR_BENCH_COMPILER_ID "")
endif()

if(NOT DEFINED FLTX_F256_EXPR_BENCH_REPEATS)
    set(FLTX_F256_EXPR_BENCH_REPEATS 3)
endif()

if(NOT DEFINED FLTX_F256_EXPR_BENCH_COUNT)
    set(FLTX_F256_EXPR_BENCH_COUNT 1000)
endif()

set(FLTX_F256_EXPR_BENCH_SOURCE_DIR "${FLTX_F256_EXPR_BENCH_BINARY_DIR}/generated")
set(FLTX_F256_EXPR_BENCH_OBJECT_DIR "${FLTX_F256_EXPR_BENCH_BINARY_DIR}/objects")
file(MAKE_DIRECTORY "${FLTX_F256_EXPR_BENCH_SOURCE_DIR}")
file(MAKE_DIRECTORY "${FLTX_F256_EXPR_BENCH_OBJECT_DIR}")

include("${CMAKE_CURRENT_LIST_DIR}/f256_expression_bench_cases.cmake")

function(fltx_f256_write_source name text)
    file(WRITE "${FLTX_F256_EXPR_BENCH_SOURCE_DIR}/${name}" "${text}")
endfunction()

function(fltx_f256_kernel_expression out_var kernel_id)
    if(kernel_id STREQUAL "pass")
        set(expr "a")
    elseif(kernel_id STREQUAL "add")
        set(expr "a + b")
    elseif(kernel_id STREQUAL "sub")
        set(expr "a - b")
    elseif(kernel_id STREQUAL "mul")
        set(expr "a * b")
    elseif(kernel_id STREQUAL "div")
        set(expr "a / b")
    elseif(kernel_id STREQUAL "mul_add")
        set(expr "a * b + c")
    elseif(kernel_id STREQUAL "mul_sub")
        set(expr "a * b - c")
    elseif(kernel_id STREQUAL "value_sub_mul")
        set(expr "c - a * b")
    elseif(kernel_id STREQUAL "mul_add_mul")
        set(expr "a * b + c * d")
    elseif(kernel_id STREQUAL "mul_sub_mul")
        set(expr "a * b - c * d")
    elseif(kernel_id STREQUAL "mul_add_mul_add")
        set(expr "a * b + c * d + e")
    elseif(kernel_id STREQUAL "mul_add_mul_sub")
        set(expr "a * b + c * d - e")
    elseif(kernel_id STREQUAL "mul_sub_mul_add")
        set(expr "a * b - c * d + e")
    elseif(kernel_id STREQUAL "mul_sub_mul_sub")
        set(expr "a * b - c * d - e")
    elseif(kernel_id STREQUAL "mul_add_add")
        set(expr "a * b + c + d")
    elseif(kernel_id STREQUAL "mul_add_sub")
        set(expr "a * b + c - d")
    elseif(kernel_id STREQUAL "mul_sub_add")
        set(expr "a * b - c + d")
    elseif(kernel_id STREQUAL "mul_sub_sub")
        set(expr "a * b - c - d")
    elseif(kernel_id STREQUAL "three_products")
        set(expr "a * b + c * d + e * f")
    elseif(kernel_id STREQUAL "four_products")
        set(expr "a * b + c * d + e * f + g * h")
    elseif(kernel_id STREQUAL "add_mul_double")
        set(expr "a + b * s")
    elseif(kernel_id STREQUAL "sub_mul_double")
        set(expr "a - b * s")
    elseif(kernel_id STREQUAL "mul_double_sub")
        set(expr "a * s - b")
    elseif(kernel_id STREQUAL "mul_double_add_mul_double")
        set(expr "a * s + b * t")
    elseif(kernel_id STREQUAL "mul_double_add_mul_double_add")
        set(expr "a * s + b * t + c")
    elseif(kernel_id STREQUAL "mul_add_div")
        set(expr "(a * b + c) / d")
    elseif(kernel_id STREQUAL "mul_sub_div")
        set(expr "(a * b - c) / d")
    elseif(kernel_id STREQUAL "value_sub_mul_div")
        set(expr "(c - a * b) / d")
    elseif(kernel_id STREQUAL "mul_add_mul_div")
        set(expr "(a * b + c * d) / e")
    elseif(kernel_id STREQUAL "mul_sub_mul_div")
        set(expr "(a * b - c * d) / e")
    elseif(kernel_id STREQUAL "add_add_sub_div")
        set(expr "(a + b - c) / d")
    elseif(kernel_id STREQUAL "add_sub_sub_div")
        set(expr "(a - b - c) / d")
    elseif(kernel_id STREQUAL "add_mul_double_div")
        set(expr "(a + b * s) / d")
    elseif(kernel_id STREQUAL "sub_mul_double_div")
        set(expr "(a - b * s) / d")
    elseif(kernel_id STREQUAL "mul_double_sub_div")
        set(expr "(a * s - b) / d")
    elseif(kernel_id STREQUAL "div_add_double")
        set(expr "a / (b + s)")
    elseif(kernel_id STREQUAL "div_double_sub")
        set(expr "a / (s - b)")
    elseif(kernel_id STREQUAL "mul_add_div_add_double")
        set(expr "(a * b + c) / (d + s)")
    elseif(kernel_id STREQUAL "mul_sub_div_add_double")
        set(expr "(a * b - c) / (d + s)")
    elseif(kernel_id STREQUAL "value_sub_mul_div_add_double")
        set(expr "(c - a * b) / (d + s)")
    elseif(kernel_id STREQUAL "mul_add_mul_div_add_double")
        set(expr "(a * b + c * d) / (e + s)")
    elseif(kernel_id STREQUAL "mul_sub_mul_div_add_double")
        set(expr "(a * b - c * d) / (e + s)")
    elseif(kernel_id STREQUAL "add_add_add_div_add_double")
        set(expr "(a + b + c) / (d + s)")
    elseif(kernel_id STREQUAL "add_sub_add_div_add_double")
        set(expr "(a - b + c) / (d + s)")
    elseif(kernel_id STREQUAL "add_add_sub_div_add_double")
        set(expr "(a + b - c) / (d + s)")
    elseif(kernel_id STREQUAL "add_sub_sub_div_add_double")
        set(expr "(a - b - c) / (d + s)")
    elseif(kernel_id STREQUAL "add_mul_double_div_add_double")
        set(expr "(a + b * s) / (d + t)")
    elseif(kernel_id STREQUAL "sub_mul_double_div_add_double")
        set(expr "(a - b * s) / (d + t)")
    elseif(kernel_id STREQUAL "mul_double_sub_div_add_double")
        set(expr "(a * s - b) / (d + t)")
    else()
        message(FATAL_ERROR "Unknown f256 expression kernel '${kernel_id}'.")
    endif()

    set(${out_var} "${expr}" PARENT_SCOPE)
endfunction()

function(fltx_f256_append_function text_var style label index expr)
    if(style STREQUAL "expr")
        set(value_type "bl::f256")
    elseif(style STREQUAL "eager")
        set(value_type "bl::f256_s")
    else()
        message(FATAL_ERROR "Unknown f256 expression benchmark style '${style}'.")
    endif()

    set(text "${${text_var}}")
    string(APPEND text
        "FLTX_EXPR_NOINLINE ${value_type} ${style}_${label}_${index}("
        "${value_type} a, ${value_type} b, ${value_type} c, ${value_type} d, "
        "${value_type} e, ${value_type} f, ${value_type} g, ${value_type} h, "
        "double s, double t) { return ${expr}; }\n"
    )
    set(${text_var} "${text}" PARENT_SCOPE)
endfunction()

function(fltx_f256_make_kernel_source name count style kernel_id)
    set(text "#include \"fltx/f256.h\"\n\n")
    string(APPEND text "#if defined(_MSC_VER)\n")
    string(APPEND text "#define FLTX_EXPR_NOINLINE __declspec(noinline)\n")
    string(APPEND text "#elif defined(__GNUC__) || defined(__clang__)\n")
    string(APPEND text "#define FLTX_EXPR_NOINLINE __attribute__((noinline))\n")
    string(APPEND text "#else\n")
    string(APPEND text "#define FLTX_EXPR_NOINLINE\n")
    string(APPEND text "#endif\n\n")
    string(APPEND text "namespace bench {\n")

    math(EXPR last_index "${count} - 1")
    foreach(i RANGE 0 ${last_index})
        if(kernel_id STREQUAL "all_supported_fused_kernels")
            list(LENGTH FLTX_F256_EXPR_KERNEL_IDS kernel_count)
            math(EXPR kernel_index "${i} % ${kernel_count}")
            list(GET FLTX_F256_EXPR_KERNEL_IDS "${kernel_index}" selected_kernel)
        else()
            set(selected_kernel "${kernel_id}")
        endif()

        fltx_f256_kernel_expression(expr "${selected_kernel}")
        fltx_f256_append_function(text "${style}" "${kernel_id}" "${i}" "${expr}")
    endforeach()

    string(APPEND text "}\n")
    fltx_f256_write_source("${name}" "${text}")
endfunction()

fltx_f256_write_source("empty.cpp" "int main() { return 0; }\n")
fltx_f256_write_source("inc_f128.cpp" "#include \"fltx/f128.h\"\nint main() { return 0; }\n")
fltx_f256_write_source("inc_f256.cpp" "#include \"fltx/f256.h\"\nint main() { return 0; }\n")
fltx_f256_write_source("inc_f128_math.cpp" "#include \"fltx/f128_math.h\"\nint main() { return 0; }\n")
fltx_f256_write_source("inc_f256_math.cpp" "#include \"fltx/f256_math.h\"\nint main() { return 0; }\n")
fltx_f256_write_source("inc_fltx_math.cpp" "#include \"fltx/math.h\"\nint main() { return 0; }\n")
fltx_f256_write_source("inc_f256_io.cpp" "#include \"fltx/f256_io.h\"\nint main() { return 0; }\n")

foreach(case_id IN LISTS FLTX_F256_EXPR_BENCH_STRESS_CASES)
    fltx_f256_make_kernel_source("expr_${case_id}_${FLTX_F256_EXPR_BENCH_COUNT}.cpp" "${FLTX_F256_EXPR_BENCH_COUNT}" "expr" "${case_id}")
    fltx_f256_make_kernel_source("eager_${case_id}_${FLTX_F256_EXPR_BENCH_COUNT}.cpp" "${FLTX_F256_EXPR_BENCH_COUNT}" "eager" "${case_id}")
endforeach()

function(fltx_f256_timestamp_ms out_var)
    string(TIMESTAMP seconds "%s")
    string(TIMESTAMP micros "%f")
    math(EXPR millis "${seconds} * 1000 + ${micros} / 1000")
    set(${out_var} "${millis}" PARENT_SCOPE)
endfunction()

function(fltx_f256_sanitize out_var text)
    string(MAKE_C_IDENTIFIER "${text}" sanitized)
    set(${out_var} "${sanitized}" PARENT_SCOPE)
endfunction()

function(fltx_f256_append_common_compile_args out_var)
    set(args)

    if(FLTX_F256_EXPR_BENCH_COMPILER_ID STREQUAL "MSVC")
        list(APPEND args
            /nologo
            /std:c++latest
            /EHsc
            /bigobj
            /I "${FLTX_F256_EXPR_BENCH_INCLUDE_DIR}"
            /D_CRT_SECURE_NO_WARNINGS
        )
    else()
        list(APPEND args
            -std=c++23
            -I "${FLTX_F256_EXPR_BENCH_INCLUDE_DIR}"
            -D_CRT_SECURE_NO_WARNINGS
        )
    endif()

    set(${out_var} "${args}" PARENT_SCOPE)
endfunction()

function(fltx_f256_compile_args out_var case_name source_file mode variant repeat_index)
    fltx_f256_sanitize(safe_case "${case_name}")
    fltx_f256_sanitize(safe_mode "${mode}")
    fltx_f256_sanitize(safe_variant "${variant}")
    set(object_path "${FLTX_F256_EXPR_BENCH_OBJECT_DIR}/${safe_case}_${safe_mode}_${safe_variant}_${repeat_index}.obj")
    set(source_path "${FLTX_F256_EXPR_BENCH_SOURCE_DIR}/${source_file}")

    fltx_f256_append_common_compile_args(args)

    if(FLTX_F256_EXPR_BENCH_COMPILER_ID STREQUAL "MSVC")
        if(mode STREQUAL "syntax")
            list(APPEND args /Zs "${source_path}")
        elseif(mode STREQUAL "Od")
            list(APPEND args /c /Od /Ob0 "/Fo${object_path}" "${source_path}")
        elseif(mode STREQUAL "O2")
            if(variant STREQUAL "Ob0")
                list(APPEND args /c /O2 /Ob0 /DNDEBUG "/Fo${object_path}" "${source_path}")
            else()
                list(APPEND args /c /O2 /Ob2 /DNDEBUG "/Fo${object_path}" "${source_path}")
            endif()
        else()
            message(FATAL_ERROR "Unknown compile mode '${mode}'.")
        endif()
    else()
        if(mode STREQUAL "syntax")
            list(APPEND args -fsyntax-only "${source_path}")
        elseif(mode STREQUAL "Od")
            list(APPEND args -c -O0 -o "${object_path}" "${source_path}")
        elseif(mode STREQUAL "O2")
            if(variant STREQUAL "Ob0")
                list(APPEND args -c -O2 -fno-inline -DNDEBUG -o "${object_path}" "${source_path}")
            else()
                list(APPEND args -c -O2 -DNDEBUG -o "${object_path}" "${source_path}")
            endif()
        else()
            message(FATAL_ERROR "Unknown compile mode '${mode}'.")
        endif()
    endif()

    set(${out_var} "${args}" PARENT_SCOPE)
endfunction()

function(fltx_f256_run_compiler elapsed_ms_var)
    fltx_f256_timestamp_ms(start_ms)
    execute_process(
        COMMAND "${FLTX_F256_EXPR_BENCH_COMPILER}" ${ARGN}
        RESULT_VARIABLE result
        OUTPUT_VARIABLE stdout
        ERROR_VARIABLE stderr
    )
    fltx_f256_timestamp_ms(end_ms)
    math(EXPR elapsed_ms "${end_ms} - ${start_ms}")

    if(NOT result EQUAL 0)
        message(FATAL_ERROR
            "Compiler failed with exit code ${result}\n"
            "Command: ${FLTX_F256_EXPR_BENCH_COMPILER};${ARGN}\n"
            "stdout:\n${stdout}\n"
            "stderr:\n${stderr}"
        )
    endif()

    set(${elapsed_ms_var} "${elapsed_ms}" PARENT_SCOPE)
endfunction()

function(fltx_f256_median out_var)
    set(values ${ARGN})
    list(SORT values COMPARE NATURAL ORDER ASCENDING)
    list(LENGTH values count)
    math(EXPR middle "${count} / 2")
    list(GET values "${middle}" median)
    set(${out_var} "${median}" PARENT_SCOPE)
endfunction()

function(fltx_f256_csv_escape out_var text)
    string(REPLACE "\"" "\"\"" escaped "${text}")
    set(${out_var} "\"${escaped}\"" PARENT_SCOPE)
endfunction()

function(fltx_f256_add_csv_row category case_name style source_file mode variant median count runs)
    if(count GREATER 0)
        math(EXPR us_per_function "(${median} * 1000 + ${count} / 2) / ${count}")
    else()
        set(us_per_function "")
    endif()

    fltx_f256_csv_escape(csv_category "${category}")
    fltx_f256_csv_escape(csv_case "${case_name}")
    fltx_f256_csv_escape(csv_style "${style}")
    fltx_f256_csv_escape(csv_file "${source_file}")
    fltx_f256_csv_escape(csv_mode "${mode}")
    fltx_f256_csv_escape(csv_variant "${variant}")
    fltx_f256_csv_escape(csv_runs "${runs}")
    string(APPEND FLTX_F256_EXPR_BENCH_CSV_ROWS
        "${csv_category},${csv_case},${csv_style},${csv_file},${csv_mode},${csv_variant},${median},${count},${us_per_function},${csv_runs}\n"
    )
    set(FLTX_F256_EXPR_BENCH_CSV_ROWS "${FLTX_F256_EXPR_BENCH_CSV_ROWS}" PARENT_SCOPE)
endfunction()

function(fltx_f256_measure_case median_var category case_name style source_file mode variant count)
    set(times)

    math(EXPR last_repeat "${FLTX_F256_EXPR_BENCH_REPEATS} - 1")
    foreach(repeat_index RANGE 0 ${last_repeat})
        fltx_f256_compile_args(args "${category} ${case_name} ${style}" "${source_file}" "${mode}" "${variant}" "${repeat_index}")
        fltx_f256_run_compiler(elapsed_ms ${args})
        list(APPEND times "${elapsed_ms}")
    endforeach()

    fltx_f256_median(median ${times})
    list(JOIN times "|" joined_times)
    fltx_f256_add_csv_row("${category}" "${case_name}" "${style}" "${source_file}" "${mode}" "${variant}" "${median}" "${count}" "${joined_times}")
    set(FLTX_F256_EXPR_BENCH_CSV_ROWS "${FLTX_F256_EXPR_BENCH_CSV_ROWS}" PARENT_SCOPE)
    set(${median_var} "${median}" PARENT_SCOPE)
endfunction()

function(fltx_f256_measure_expr_vs_eager case_id mode variant)
    set(expr_file "expr_${case_id}_${FLTX_F256_EXPR_BENCH_COUNT}.cpp")
    set(eager_file "eager_${case_id}_${FLTX_F256_EXPR_BENCH_COUNT}.cpp")

    fltx_f256_measure_case(eager_ms "expression-kernel" "${case_id}" "eager f256_s" "${eager_file}" "${mode}" "${variant}" "${FLTX_F256_EXPR_BENCH_COUNT}")
    fltx_f256_measure_case(expr_ms "expression-kernel" "${case_id}" "expr f256" "${expr_file}" "${mode}" "${variant}" "${FLTX_F256_EXPR_BENCH_COUNT}")

    math(EXPR delta_ms "${expr_ms} - ${eager_ms}")
    if(eager_ms GREATER 0)
        math(EXPR ratio_percent "(${expr_ms} * 100 + ${eager_ms} / 2) / ${eager_ms}")
    else()
        set(ratio_percent "n/a")
    endif()

    message(STATUS
        "kernel ${case_id} [${mode}, ${variant}]: "
        "expr ${expr_ms} ms vs eager ${eager_ms} ms; "
        "expr/eager ${ratio_percent}%; delta ${delta_ms} ms"
    )
endfunction()

set(FLTX_F256_EXPR_BENCH_CSV_ROWS "")

message(STATUS "f256 expression compile benchmark compiler: ${FLTX_F256_EXPR_BENCH_COMPILER}")
message(STATUS "f256 expression compile benchmark generated sources: ${FLTX_F256_EXPR_BENCH_SOURCE_DIR}")
message(STATUS "f256 expression compile benchmark object output: ${FLTX_F256_EXPR_BENCH_OBJECT_DIR}")
message(STATUS "f256 expression compile benchmark repeats: ${FLTX_F256_EXPR_BENCH_REPEATS}")
message(STATUS "f256 expression compile benchmark functions per case: ${FLTX_F256_EXPR_BENCH_COUNT}")
message(STATUS "style 'expr f256' uses public bl::f256 expression templates")
message(STATUS "style 'eager f256_s' uses plain bl::f256_s eager arithmetic")
message(STATUS "'all_supported_fused_kernels' cycles through every generated fused-kernel expression shape")

# Warm the compiler process and filesystem caches before taking measurements.
fltx_f256_compile_args(warm_args "warm include fltx/f256.h" "inc_f256.cpp" "syntax" "default" 0)
fltx_f256_run_compiler(warm_elapsed ${warm_args})

fltx_f256_measure_case(empty_ms "include" "empty translation unit" "none" "empty.cpp" "syntax" "default" 0)
message(STATUS "include empty translation unit [syntax, default]: ${empty_ms} ms")

set(include_case_names
    "fltx/f128.h"
    "fltx/f256.h"
    "fltx/f128_math.h"
    "fltx/f256_math.h"
    "fltx/math.h"
    "fltx/f256_io.h"
)
set(include_case_files
    "inc_f128.cpp"
    "inc_f256.cpp"
    "inc_f128_math.cpp"
    "inc_f256_math.cpp"
    "inc_fltx_math.cpp"
    "inc_f256_io.cpp"
)

list(LENGTH include_case_names include_case_count)
math(EXPR include_case_last "${include_case_count} - 1")

foreach(include_case_index RANGE 0 ${include_case_last})
    list(GET include_case_names "${include_case_index}" include_name)
    list(GET include_case_files "${include_case_index}" include_file)
    fltx_f256_measure_case(include_ms "include" "${include_name}" "none" "${include_file}" "syntax" "default" 0)
    math(EXPR include_delta "${include_ms} - ${empty_ms}")
    message(STATUS "include ${include_name} [syntax, default]: ${include_ms} ms; delta vs empty ${include_delta} ms")
endforeach()

foreach(case_id IN LISTS FLTX_F256_EXPR_BENCH_STRESS_CASES)
    fltx_f256_measure_expr_vs_eager("${case_id}" "syntax" "default")
    fltx_f256_measure_expr_vs_eager("${case_id}" "O2" "Ob0")
    fltx_f256_measure_expr_vs_eager("${case_id}" "O2" "default")
endforeach()

set(csv_path "${FLTX_F256_EXPR_BENCH_BINARY_DIR}/f256_expression_compile_bench.csv")
file(WRITE "${csv_path}" "category,case,style,file,mode,variant,median_ms,function_count,us_per_function,runs_ms\n${FLTX_F256_EXPR_BENCH_CSV_ROWS}")
message(STATUS "f256 expression compile benchmark CSV: ${csv_path}")
