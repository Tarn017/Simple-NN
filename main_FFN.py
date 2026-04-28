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
