from pylearn.tools.shortcuts import buildNetwork
from pylearn.structure import SoftmaxLayer
from pylearn.structure import TanhLayer
from pylearn.datasets import SupervisedDataSet
from pylearn.supervised.trainers import BackpropTrainer
from pylearn.tools.xml.networkwriter import NetworkWriter
from pylearn.tools.xml.networkreader import NetworkReader
import matplotlib.pyplot as plt

#net = buildNetwork(2, 3, 1, hiddenclass=TanhLayer, outclass=SoftmaxLayer, bias=True)
net = buildNetwork(2, 3, 1)
y = net.activate([2, 1])

ds = SupervisedDataSet(2, 1)
ds.addSample((0, 0), (0,))
ds.addSample((0, 1), (1,))
ds.addSample((1, 0), (1,))
ds.addSample((1, 1), (0,))
print(ds)

trainer = BackpropTrainer(net)
trnerr, valerr = trainer.trainUntilConvergence(dataset=ds, maxEpochs=100)
plt.plot(trnerr, 'b', valerr, 'r')
plt.show()
