import numpy as np
#print(np.__version__) print version
a = np.array([1,2,3]) #3 columns
print(a) # imprimir a
print(a.shape) # array shape
print(a.dtype) # tipo de los valores de a
print(a.ndim) # dimension de a
print(a.size) # imprimir tamaño de a 
print(a.itemsize) # bytes of the value
print(a[0]) # primer elemento de la lista
a[0]=10 # modifico el primer elemento del array
print(a) 
b = a * np.array([2,0,2]) # producto entre matrices 
print(b)
c = a + np.array([4]) # sumar un valor c/valor del array
# array no permite el metodo append
print(c)
d = np.sqrt(a) # raiz cuadrada de un arrays
print(d)
e = np.log(a) # logaritmo de un array
print(e)
f = np.dot(a,b) # producto punto
print(f)
#g = np.random.randn(1000) # generar un array de 1000 valores
#print(g)
h = np.array([[1,2,3],[4,5,6]]) #multidimensional array
print(h,h.shape)
print(h[1][2])
print(h,h.T) # Transpose 
i = np.array([[1,2],[3,4]])
print(i)
print(np.linalg.inv(i)) # inverse of an array
print(np.linalg.det(i)) # determinant of an array
print(np.diag(i)) # diagonal of an array

