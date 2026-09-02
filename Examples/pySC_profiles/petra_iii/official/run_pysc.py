from pySC import generate_SC

SC = generate_SC('petra3_conf.yaml')
SC.start_server()
# bad_quad = SC.magnet_arrays['bad_quad'][0]
# 
# error = SC.magnet_settings.magnets[bad_quad]._links[0].error.factor
# 
# print(f"Relative error of {bad_quad} is {(error-1)*100:.2f}%")