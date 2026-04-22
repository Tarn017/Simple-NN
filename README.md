# Simple-NN

**Vorgehen: **  
1. Ladet zunächst das Repository herunter. Geht dafür oben auf der Seite auf das grüne Feld "Code" und anschließend "Download ZIP":

![Download Screenshot](https://github.com/Tarn017/Simple-NN/blob/main/Assets/Repo_download.png)

3. Kopiert den Ordner "tiere400", "requirements_class.txt`, das Skript "NeuralNetwork.py" und "main_FFN.py" euer Python-Projekt.  
4. Öffnet nun das Skript `main_FFN.py`. Zunächst müssen, falls noch nichtgeschehen, die nötigen Bibliotheken herunter geladen werden. Geht dafür unten auf das Terminal und gebt ein `pip install -r requirements_class.txt`:

![requirements Screenshot](https://github.com/Tarn017/Simple-NN/blob/main/Assets/requirements.png)

**Training: **

Für das Training wird nun die folgende Funktion genutzt:  
`FFN(train_path, model_name, resize, epochs, dataset_size, fully_layers, lr=, dec_lr, droprate, augmentation)`  
1. *train_path* entspricht dem Ordner auf den ihr das Netz trainieren wollt
2. *epochs* entspricht der Anzahl an Epochen, die das Netz trainiert werden soll
3. *lr* entspricht der Lernrate
4. *conv_filters* hat die Form [x,y,z,...], wobei x der Anzahl an convolutional Filtern in der 1. Schicht entsprich, y der Anzahl in der 2., usw. Die GEsamtzahl an Schichten wird somit ebenfalls hier bestimmt.
5. *fully_layers* hat die Form [x,y,z,...], wobei x der Anzahl an Neuronen in der 1. voll verbundenen Schicht entspricht, usw.
6. *resize* hat die form (höhe,breite), wobei beide Angaben in Anzahl Pixel gemacht werden. Quadratische Angaben werden bevorzugt
7. *model_name* gibt eurem Modell einen Namen. Beachte: Existiert bereits eines mit demselben Namen, wird das alte überschrieben.
8. *train_split* bspw. 0.8 bedeutet, dass 80% der Bilder fürs Training und 20% für Validation genutzt werden. Bei 1 werden alle Daten für das Training genutzt.
9. *droprate* (optional) bspw 0.2 bedeutet, dass jedes Neuron mit einer Wahrscheinlichkeit von 20% deaktiviert wird.
10. *augmentation* (optional) hat die Form [flip, rotate, brightness, contrast, saturation]. Jeder dieser Werte muss zwischen 0 und 1 liegen. *flip* ist die Wahrscheinlichkeit, dass ein Bild gespiegelt wird, *rotate* gibt die Stärke einer zufälligen Rotation an (1 für stark), *brightness, contrast, saturation* geben an wie stark Helligkeit, Kontrast und Sättigung maximal verändert werden.
11. *dec_lr* (optional) wirg genutzt, falls die Lernrate während des Trainigs abnehmen soll. Der Wert der für diesen Parameter angegeben wird, entspricht der Lernrate der letzten Epoche.

Hier ein Beispiel. Wichtig ist, dass Anführungszeichen übernommen werden, dort wo sie gebraucht werden und der Datentyp für jedem Parameter richtig gewählt ist: 
```python
from project import CNN

if __name__ == "__main__":
    CNN(
        train_path="zml_klass",
        epochs=15,
        lr=0.001,
        conv_filters=[16, 32, 64, 128],
        fully_layers=[256],
        resize=(128, 128),
        model_name='peter',
        train_split=0.9,
        droprate=0,
        augmentation=[0, 0, 0, 0, 0],
        dec_lr=0.001
    )
```
