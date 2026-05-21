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

function(fltx_f256_write_source name text)
    file(WRITE "${FLTX_F256_EXPR_BENCH_SOURCE_DIR}/${name}" "${text}")
endfunction()

function(fltx_f256_make_expr_source name count kind)
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
        if(kind STREQUAL "pass")
            set(expr "return a;")
        elseif(kind STREQUAL "add")
            set(expr "return a + b;")
        elseif(kind STREQUAL "muladdmul")
            set(expr "return a * b + c * d;")
        elseif(kind STREQUAL "muladdmul_direct")
            set(expr "return bl::f256{ bl::detail::_f256_runtime::mul_add_mul(static_cast<const bl::f256_s&>(a), static_cast<const bl::f256_s&>(b), static_cast<const bl::f256_s&>(c), static_cast<const bl::f256_s&>(d)) };")
        elseif(kind STREQUAL "eager_steps")
            set(expr "bl::f256 ab = a * b; bl::f256 cd = c * d; return ab + cd;")
        elseif(kind STREQUAL "mixed")
            math(EXPR variant "${i} % 8")
            if(variant EQUAL 0)
                set(expr "return a * b + c * d;")
            elseif(variant EQUAL 1)
                set(expr "return (a * b + c) / (d + 1.0);")
            elseif(variant EQUAL 2)
                set(expr "return a * 2.0 + b * 3.0 + c;")
            elseif(variant EQUAL 3)
                set(expr "return (a + b + c) / d;")
            elseif(variant EQUAL 4)
                set(expr "return a - b * c;")
            elseif(variant EQUAL 5)
                set(expr "return (a * b - c * d) / (a - 2.0);")
            elseif(variant EQUAL 6)
                set(expr "return (a * 0.5 + b * 0.25) - c;")
            else()
                set(expr "return (a * b + c * d + a * c + b * d);")
            endif()
        else()
            message(FATAL_ERROR "Unknown expression source kind '${kind}'.")
        endif()

        string(APPEND text "FLTX_EXPR_NOINLINE bl::f256 ${kind}_${i}(bl::f256 a, bl::f256 b, bl::f256 c, bl::f256 d) { ${expr} }\n")
    endforeach()

    string(APPEND text "}\n")
    fltx_f256_write_source("${name}" "${text}")
endfunction()

fltx_f256_write_source("empty.cpp" "int main() { return 0; }\n")
fltx_f256_write_source("inc_f128.cpp" "#include \"fltx/f128.h\"\nint main() { return 0; }\n")
fltx_f256_write_source("inc_f256.cpp" "#include \"fltx/f256.h\"\nint main() { return 0; }\n")
fltx_f256_write_source("inc_f128_math.cpp" "#include \"fltx/f128/math.h\"\nint main() { return 0; }\n")
fltx_f256_write_source("inc_f256_math.cpp" "#include \"fltx/f256/math.h\"\nint main() { return 0; }\n")
fltx_f256_write_source("inc_fltx_math.cpp" "#include \"fltx/math.h\"\nint main() { return 0; }\n")
fltx_f256_write_source("inc_f256_io.cpp" "#include \"fltx/f256/io.h\"\nint main() { return 0; }\n")

fltx_f256_make_expr_source("expr_pass_${FLTX_F256_EXPR_BENCH_COUNT}.cpp" "${FLTX_F256_EXPR_BENCH_COUNT}" "pass")
fltx_f256_make_expr_source("expr_add_${FLTX_F256_EXPR_BENCH_COUNT}.cpp" "${FLTX_F256_EXPR_BENCH_COUNT}" "add")
fltx_f256_make_expr_source("expr_muladdmul_100.cpp" 100 "muladdmul")
fltx_f256_make_expr_source("expr_muladdmul_${FLTX_F256_EXPR_BENCH_COUNT}.cpp" "${FLTX_F256_EXPR_BENCH_COUNT}" "muladdmul")
fltx_f256_make_expr_source("expr_muladdmul_direct_${FLTX_F256_EXPR_BENCH_COUNT}.cpp" "${FLTX_F256_EXPR_BENCH_COUNT}" "muladdmul_direct")
fltx_f256_make_expr_source("expr_eager_steps_${FLTX_F256_EXPR_BENCH_COUNT}.cpp" "${FLTX_F256_EXPR_BENCH_COUNT}" "eager_steps")
fltx_f256_make_expr_source("expr_mixed_${FLTX_F256_EXPR_BENCH_COUNT}.cpp" "${FLTX_F256_EXPR_BENCH_COUNT}" "mixed")

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

function(fltx_f256_add_csv_row case_name source_file mode variant median runs)
    fltx_f256_csv_escape(csv_case "${case_name}")
    fltx_f256_csv_escape(csv_file "${source_file}")
    fltx_f256_csv_escape(csv_mode "${mode}")
    fltx_f256_csv_escape(csv_variant "${variant}")
    fltx_f256_csv_escape(csv_runs "${runs}")
    string(APPEND FLTX_F256_EXPR_BENCH_CSV_ROWS "${csv_case},${csv_file},${csv_mode},${csv_variant},${median},${csv_runs}\n")
    set(FLTX_F256_EXPR_BENCH_CSV_ROWS "${FLTX_F256_EXPR_BENCH_CSV_ROWS}" PARENT_SCOPE)
endfunction()

function(fltx_f256_measure_case case_name source_file mode variant)
    set(times)

    math(EXPR last_repeat "${FLTX_F256_EXPR_BENCH_REPEATS} - 1")
    foreach(repeat_index RANGE 0 ${last_repeat})
        fltx_f256_compile_args(args "${case_name}" "${source_file}" "${mode}" "${variant}" "${repeat_index}")
        fltx_f256_run_compiler(elapsed_ms ${args})
        list(APPEND times "${elapsed_ms}")
    endforeach()

    fltx_f256_median(median ${times})
    list(JOIN times "|" joined_times)
    fltx_f256_add_csv_row("${case_name}" "${source_file}" "${mode}" "${variant}" "${median}" "${joined_times}")
    set(FLTX_F256_EXPR_BENCH_CSV_ROWS "${FLTX_F256_EXPR_BENCH_CSV_ROWS}" PARENT_SCOPE)
    message(STATUS "${case_name} [${mode}, ${variant}]: median ${median} ms; runs ${joined_times}")
endfunction()

set(FLTX_F256_EXPR_BENCH_CSV_ROWS "")

message(STATUS "f256 expression compile benchmark compiler: ${FLTX_F256_EXPR_BENCH_COMPILER}")
message(STATUS "f256 expression compile benchmark generated sources: ${FLTX_F256_EXPR_BENCH_SOURCE_DIR}")
message(STATUS "f256 expression compile benchmark repeats: ${FLTX_F256_EXPR_BENCH_REPEATS}")
message(STATUS "f256 expression compile benchmark count: ${FLTX_F256_EXPR_BENCH_COUNT}")

# Warm the compiler process and filesystem caches before taking measurements.
fltx_f256_compile_args(warm_args "warm include fltx/f256.h" "inc_f256.cpp" "syntax" "default" 0)
fltx_f256_run_compiler(warm_elapsed ${warm_args})

fltx_f256_measure_case("empty" "empty.cpp" "syntax" "default")
fltx_f256_measure_case("include fltx/f128.h" "inc_f128.cpp" "syntax" "default")
fltx_f256_measure_case("include fltx/f256.h" "inc_f256.cpp" "syntax" "default")
fltx_f256_measure_case("include fltx/f128/math.h" "inc_f128_math.cpp" "syntax" "default")
fltx_f256_measure_case("include fltx/f256/math.h" "inc_f256_math.cpp" "syntax" "default")
fltx_f256_measure_case("include fltx/math.h" "inc_fltx_math.cpp" "syntax" "default")
fltx_f256_measure_case("include fltx/f256/io.h" "inc_f256_io.cpp" "syntax" "default")

set(large_count "${FLTX_F256_EXPR_BENCH_COUNT}")
set(large_case_names
    "${large_count} pass"
    "${large_count} add"
    "100 muladdmul"
    "${large_count} muladdmul"
    "${large_count} direct runtime"
    "${large_count} eager steps"
    "${large_count} mixed"
)
set(large_case_sources
    "expr_pass_${large_count}.cpp"
    "expr_add_${large_count}.cpp"
    "expr_muladdmul_100.cpp"
    "expr_muladdmul_${large_count}.cpp"
    "expr_muladdmul_direct_${large_count}.cpp"
    "expr_eager_steps_${large_count}.cpp"
    "expr_mixed_${large_count}.cpp"
)

list(LENGTH large_case_names large_case_count)
math(EXPR large_case_last "${large_case_count} - 1")
foreach(case_index RANGE 0 ${large_case_last})
    list(GET large_case_names "${case_index}" case_name)
    list(GET large_case_sources "${case_index}" source_file)
    fltx_f256_measure_case("${case_name}" "${source_file}" "syntax" "default")
    fltx_f256_measure_case("${case_name}" "${source_file}" "Od" "default")
    fltx_f256_measure_case("${case_name}" "${source_file}" "O2" "default")
endforeach()

set(variant_case_names
    "${large_count} add"
    "${large_count} muladdmul"
    "${large_count} direct runtime"
    "${large_count} eager steps"
    "${large_count} mixed"
)
set(variant_case_sources
    "expr_add_${large_count}.cpp"
    "expr_muladdmul_${large_count}.cpp"
    "expr_muladdmul_direct_${large_count}.cpp"
    "expr_eager_steps_${large_count}.cpp"
    "expr_mixed_${large_count}.cpp"
)

list(LENGTH variant_case_names variant_case_count)
math(EXPR variant_case_last "${variant_case_count} - 1")
foreach(case_index RANGE 0 ${variant_case_last})
    list(GET variant_case_names "${case_index}" case_name)
    list(GET variant_case_sources "${case_index}" source_file)
    fltx_f256_measure_case("${case_name}" "${source_file}" "O2" "Ob0")
endforeach()

set(csv_path "${FLTX_F256_EXPR_BENCH_BINARY_DIR}/f256_expression_compile_bench.csv")
file(WRITE "${csv_path}" "case,file,mode,variant,median_ms,runs_ms\n${FLTX_F256_EXPR_BENCH_CSV_ROWS}")
message(STATUS "f256 expression compile benchmark CSV: ${csv_path}")
