from predictor import predict

demo_tx = {'V17':-2.5,'V14':-3.0,'V12':-4.0,'V10':-5.0,'V11':-2.0}
result = predict(demo_tx)
print(result)
