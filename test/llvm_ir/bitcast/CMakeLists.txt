if(ENABLE_TESTING)
  add_mli_test(name_base BITCAST_INTTOFP subdir itofp filename_base test_itofp mli_args "--args=1069547520" regex ".+result: .+1.5")
  add_mli_test(name_base BITCAST_FPTOINT subdir fptoi filename_base test_fptoi mli_args "--args=1.5" regex ".+result: .+1069547520" )
endif()
