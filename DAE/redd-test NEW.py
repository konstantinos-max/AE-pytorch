from __future__ import print_function, division
import time

from matplotlib import rcParams
import matplotlib.pyplot as plt

from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
from vaedisaggregator_pytorch5 import DAEDisaggregator
import metrics

print("========== OPEN DATASETS ============")
train = DataSet('redd.h5')
test = DataSet('redd.h5')

train.set_window(end="30-4-2011")
test.set_window(start="30-4-2011")

test_building = 1
meter_key = 'fridge'  # appliance
train_elec = train.buildings[1].elec
test_elec = test.buildings[test_building].elec

train_meter = train_elec.submeters()[meter_key]
train_mains = train_elec.mains().all_meters()[0]
test_mains = test_elec.mains().all_meters()[0]
dae = DAEDisaggregator(100) # or 256 or 500


start = time.time()
print("========== TRAIN ============")
dae.train(train_mains, train_meter, epochs=10, sample_period=1)   # evala 10 anti gia 5
dae.export_model("model-redd100.h5")
end = time.time()
print("Train =", end-start, "seconds.")


print("========== DISAGGREGATE ============")
disag_filename = 'disag-out.h5'
output = HDFDataStore(disag_filename, 'w')
dae.disaggregate(test_mains, output, train_meter, sample_period=1)   # train_meter prin kai xtyphse me test_mains meta
output.close()

print("========== RESULTS ============")
result = DataSet(disag_filename)
res_elec = result.buildings[test_building].elec
print(res_elec)
rpaf = metrics.recall_precision_accuracy_f1(res_elec[meter_key], test_elec[meter_key])
print("============ Recall: {}".format(rpaf[0]))
print("============ Precision: {}".format(rpaf[1]))
print("============ Accuracy: {}".format(rpaf[2]))
print("============ F1 Score: {}".format(rpaf[3])) ### sxolio

print("============ Relative error in total energy: {}".format(metrics.relative_error_total_energy(res_elec[meter_key], test_elec[meter_key])))
print("============ Mean absolute error(in Watts): {}".format(metrics.mean_absolute_error(res_elec[meter_key], test_elec[meter_key])))

res_elec[meter_key].plot()
test_elec[meter_key].plot()
plt.show()
