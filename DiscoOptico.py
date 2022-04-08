from skimage import io, measure, exposure, morphology, filters
import numpy as np
import matplotlib.pyplot as plt


plt.close('all')

Color = io.imread('imagenes/01_test.tif')

rojo = Color[:, :, 1]

ventana = morphology.disk(30)
Ima_Fil = filters.rank.mean(rojo, selem=ventana)

Adap = exposure.equalize_adapthist(Ima_Fil)

fig, axs = plt.subplots(2, 2)
fig.suptitle('Procesos en el Ojo')
axs[0, 0].imshow(Color)
axs[0, 1].imshow(rojo, cmap='gray')
axs[1, 0].imshow(Ima_Fil, cmap='gray')
axs[1, 1].imshow(Adap, cmap='gray')

# Preguntar de que lado esta el disco optico para recortar la imagen.
lado = int(input('Menciona si el ojo es el izquierdo o el derecho 1=I 2=D:  '))

if (lado == 1):
    recorte = Adap[778:1557, 1:1168]
elif (lado == 2):
    recorte = Adap[778:1557, 2000:3504]
else:
    print('Respuesta no existe')

disco = morphology.disk(30)
Open = morphology.opening(recorte, disco)
Close = morphology.closing(Open, disco)

valor = np.max(Close)
if (valor < 1):
    a = 1 - valor
    Close = Close + a

Bin = (Close > 0.98).astype(int)
Label = measure.label(Bin)
Centro = measure.regionprops(Label)

for props in Centro:
    y, x = props.centroid

if (lado == 1):
    y0 = y + 778
    x0 = x
else:
    y0 = y + 778
    x0 = x + 2000

fig, axs = plt.subplots(2, 2)
fig.suptitle('Recorte y Disco')
axs[0, 0].imshow(recorte, cmap='gray')
axs[0, 1].imshow(Close,   cmap='gray')
axs[1, 0].imshow(Bin,     cmap='gray')
axs[1, 1].imshow(Color,   cmap='gray')
axs[1, 1].plot(x0, y0, '.g', markersize=5)


for ii in range(Color.shape[0]):
    for jj in range(Color.shape[1]):
        radio = ((ii - y0)**2 + (jj - x0)**2)
        if (radio < 45000):
            Color[ii, jj, :] = 0

plt.figure()
plt.imshow(Color)
