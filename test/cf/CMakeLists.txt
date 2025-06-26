if(ENABLE_TESTING)
	add_mli_test(name_base CF_ASSERT_TRUE subdir assert mli_args "--args=1" regex "Function completed")
	add_mli_test(name_base CF_ASSERT_FALSE subdir assert mli_args "--args=0" regex "assertion failed" expect_failure TRUE)
	# add_mli_test(name_base CF_BR subdir br regex "Function completed")
endif()

