def get_Xy(domain, path_to_folder="../data/OfficeHomeDataset/"):

    path = path_to_folder + domain 
    print(path)
    X = []
    y = []

    for r, d, f in os.walk(path):
        for direct in d:
            if not ".ipynb_checkpoints" in direct:
                for r, d, f in os.walk(os.path.join(path , direct)):
                    for file in f:
                        path_to_image = os.path.join(r, file)
                        if not ".ipynb_checkpoints" in path_to_image:
                            image = Image.open(path_to_image)
                            image = image.resize((224, 224), Image.ANTIALIAS)
                            image = np.array(image, dtype=int)
                            X.append(image)
                            y.append(direct)
                            
    X = np.asarray(X)
    y = np.asarray(y)
                            
    X = X.astype('float32') / 255.
    
    return X, y