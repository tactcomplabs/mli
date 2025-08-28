if(ENABLE_TESTING)
  add_mli_test(name_base MISC_BIG_INT subdir big_int mli_args "--args=18446744073709551616,1" regex "18446744073709551617")
  add_mli_test(name_base MISC_FP128 subdir fp_128 mli_args "--args=1,10000000000,1" function_name quad_form regex "-1\\.[0-9]*E\\+10, ?-1\\.[0-9]*E-10")
endif()

