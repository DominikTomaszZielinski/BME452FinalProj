import matlab.engine
eng = matlab.engine.start_matlab()
result = eng.eval("2+2", nargout=1)
print(f"MATLAB says 2+2 = {result}")
eng.quit()