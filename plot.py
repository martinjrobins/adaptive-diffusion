import numpy as np
import matplotlib.pyplot as plt
from vtk import vtkXMLUnstructuredGridReader
from vtk.util import numpy_support as VN

filenamebase = 'explicit-random'
error = []
for i in range(20):
    filename = filenamebase+'%05d.vtu'%i
    print 'reading file ',filename

    reader = vtkXMLUnstructuredGridReader()
    reader.SetFileName(filename)
    #reader.ReadAllVectorsOn()
    #reader.ReadAllScalarsOn()
    reader.Update()

    data = reader.GetOutput()

    u = VN.vtk_to_numpy(data.GetPointData().GetArray('temperature'))
    s = VN.vtk_to_numpy(data.GetPointData().GetArray('solution'))

    error.append(np.sqrt(np.sum((u-s)**2))/np.sqrt(np.sum(s**2)))

    #x = zeros(data.GetNumberOfPoints())
    #y = zeros(data.GetNumberOfPoints())
    #z = zeros(data.GetNumberOfPoints())

    #for i in range(data.GetNumberOfPoints()):
    #        x[i],y[i],z[i] = data.GetPoint(i)

plt.figure(figsize=(6,4.5))
plt.plot(range(20),error)
plt.xlabel('$n$')
plt.ylabel('$\mathcal{E}$')
plt.savefig('error.pdf')
