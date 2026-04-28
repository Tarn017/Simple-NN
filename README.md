# Simple-NN

**Vorgehen:**  
1. Ladet zunächst das Repository herunter. Geht dafür hier auf der Seite oben auf das grüne Feld "Code" und anschließend "Download ZIP":

![Download Screenshot](https://github.com/Tarn017/Simple-NN/blob/main/Assets/Repo_download.png)

3. Kopiert den Ordner "tiere400", "requirements_class.txt`, das Skript "NeuralNetwork.py" und "main_FFN.py" in euer Python-Projekt.  
4. Öffnet nun das Skript `main_FFN.py`. Zunächst müssen, falls noch nichtgeschehen, die nötigen Bibliotheken herunter geladen werden. Geht dafür unten auf das Terminal und gebt ein `pip install -r requirements_class.txt`:

![requirements Screenshot](https://github.com/Tarn017/Simple-NN/blob/main/Assets/requirements.png)

**Training:**

Für das Training wird nun die folgende Funktion genutzt:  
`FFN(train_path, model_name, resize, epochs, dataset_size, fully_layers, lr=, dec_lr, droprate, augmentation)`  
1. *train_path* entspricht dem Ordner auf den ihr das Netz trainieren wollt. Einfach 'tiere400' angeben.
2. *dataset_size* bestimmt die größe des Datensatzes. Der Wert muss sich zwischen 0 und 1 befinden. Ein Wert von 0.05 entspricht einem sehr kleinen Datensatz  und ein Wert von 0.95 einem sehr großen.
3. *epochs* entspricht der Anzahl an Epochen, die das Netz trainiert werden soll
4. *lr* entspricht der Lernrate
5. *fully_layers* hat die Form [x,y,z,...], wobei x der Anzahl an Neuronen in der 1. Schicht, y der Anzahl in der 2. Schicht, usw. entspricht. Damit wird daher auch die Anzahl an Schichten gesteuert.  
6. *resize* hat die form (höhe,breite), wobei beide Angaben in Anzahl Pixel gemacht werden. Quadratische Angaben werden bevorzugt
7. *model_name* gibt eurem Modell einen Namen. Beachte: Existiert bereits eines mit demselben Namen, wird das alte überschrieben.
8. *droprate* (optional) bspw 0.2 bedeutet, dass jedes Neuron mit einer Wahrscheinlichkeit von 20% deaktiviert wird.
9. *augmentation* (optional) hat die Form [flip, rotate, brightness, contrast, saturation]. Jeder dieser Werte muss zwischen 0 und 1 liegen. *flip* ist die Wahrscheinlichkeit, dass ein Bild gespiegelt wird, *rotate* gibt die Stärke einer zufälligen Rotation an (1 für stark), *brightness, contrast, saturation* geben an wie stark Helligkeit, Kontrast und Sättigung maximal verändert werden.
10. *dec_lr* (optional) wirg genutzt, falls die Lernrate während des Trainigs abnehmen soll. Der Wert der für diesen Parameter angegeben wird, entspricht der Lernrate der letzten Epoche.

Hier ein Beispiel. Wichtig ist, dass Anführungszeichen übernommen werden, dort wo sie gebraucht werden und der Datentyp für jedem Parameter richtig gewählt ist: 
```python
from NeuralNetwork import FFN

if __name__ == "__main__":
        FFN(
            train_path="tiere400",
            dataset_size=0.1,
            model_name='erstes_modell',
            resize=(64, 64),
            epochs=8,
            fully_layers=[500, 100, 20],
            lr=0.001,
            dec_lr=1e-5,
            droprate=0.5,
            augmentation=[0,0,0,0,0]
        )
```
