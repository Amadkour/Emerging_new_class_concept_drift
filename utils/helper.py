import numpy as np
from river.datasets import synth
import numpy as np
from river import synth
from scipy.io import arff
from strlearn.streams import StreamGenerator, ARFFParser


def realstreams():
    data = ARFFParser("datasets/coverType.arff", n_chunks=200, chunk_size=2000)
    return {
        "covertype": data,
    }

def realstreams_aea():
    data = ARFFParser("datasets/aea.arff", n_chunks=200, chunk_size=500)
    return {
        "har": data,
    }
def realstreams_har():
    data = ARFFParser("datasets/har.arff", n_chunks=100, chunk_size=500)
    return {
        "har": data,
    }


def realstreams_sensor():
    # data=ARFFParser("datasets/power.arff", n_chunks=200, chunk_size=2000)
    data = ARFFParser("datasets/sensors.arff", n_chunks=200, chunk_size=2000)
    return {
        "sensors": data,
    }


import os.path


def realstreams_stream_learning():
    # if not os.path.isfile("datasets/synthetic.arff"):
    # streaam_stream_learning_generator()
    river_hyper_plane_stream()
    data = ARFFParser("synthetic_river.arff", n_chunks=200, chunk_size=2000)
    return {
        "synthetic": data,
    }


def steaming2():
    streams = {}
    n_classes = 4
    n_features = 10
    n_chunks = 200
    drift_type = False
    spacing = 5
    stream = StreamGenerator(n_chunks=n_chunks, chunk_size=2000, n_features=n_features, n_drifts=20,
                             n_classes=n_classes,
                             y_flip=n_features * 0.7, concept_sigmoid_spacing=spacing, incremental=drift_type,
                             n_informative=5,

                             )
    if spacing == None and drift_type == True:
        pass
    else:
        streams.update({str(stream): stream})

    return streams


def steaming_river():
    streams = {}
    n_classes = 4
    n_features = 10
    n_chunks = 200
    drift_type = False
    spacing = 5
    stream = synth.LEDDrift(seed=112, noise_percentage=0.28,
                            irrelevant_features=True, n_drift_features=4)

    streams.update({str(stream): stream})

    return streams


def streaam_stream_learning_generator():
    import numpy as np
    from sklearn.datasets import make_classification

    # Set the seed for reproducibility
    np.random.seed(42)

    # Generate an initial dataset with two existing classes
    X, y = make_classification(
        n_samples=2000,
        n_features=10,
        n_informative=5,
        scale=1,
        n_classes=1,
        random_state=42
    )
    for i in range(1, 200):
        print(i)
        n_classes = 1 + i
        if n_classes * 2 > 32:
            n_informative = int(np.ceil(np.log2(n_classes * 2)))
        else:
            n_informative = 5
        try:
            X_emerging, y_emerging = make_classification(
                n_samples=2000,
                n_features=10,
                n_informative=n_informative,
                n_classes=n_classes,
                random_state=42
            )
        except:
            X_emerging, y_emerging = make_classification(
                n_samples=2000,
                n_features=10,
                n_informative=5,
                scale=1,
                n_classes=1,
                random_state=42
            )
        X = np.concatenate((X, X_emerging), axis=0)
        y = np.concatenate((y, y_emerging), axis=0)
    print(str(np.unique(y)).replace("[", "{").replace("]", "}"))
    with open('synthetic.arff', 'w') as file:
        file.write(f'''@relation synthetic
@attribute att0 numeric
@attribute att1 numeric
@attribute att2 numeric
@attribute att3 numeric
@attribute att4 numeric
@attribute att5 numeric
@attribute att6 numeric
@attribute att7 numeric
@attribute att8 numeric
@attribute att9 numeric
@attribute class %{str(np.unique(y)).replace("[", "{").replace("]", "}")}
@data
''')
        # Append the content to the file
        for i in range(len(X)):
            s = ''
            for j in range(len(X[i])):
                if (j == 9):
                    s += str(X[i][j]) + ',' + str(y[i])
                else:
                    s += str(X[i][j]) + ','
            file.write(s + '\n')
        file.close()

def synthetic_streams():
    # Variables
    distribution= [0.5,0.27, 0.03,0.22]
    n_drifts = 30

    # Prepare streams
    streams = {}

    stream = StreamGenerator(
        incremental=False,
        weights=distribution,
        n_classes=4,
        random_state=125,
        y_flip=0.05,
        concept_sigmoid_spacing=5,
        n_drifts= 30,
        chunk_size=2000,
        n_chunks=200,
        n_clusters_per_class=1,
        n_features=8,
        n_informative=8,
        n_redundant=0,
        n_repeated=0,
    )

    streams.update({str(stream): stream})

    return streams


def timestream(chunk_size):
    # Variables
    distributions = [[0.80, 0.20]]
    label_noises = [
        0.01,
    ]
    incremental = [False]
    ccs = [None]
    n_drifts = 1

    # Prepare streams
    streams = {}
    for drift_type in incremental:
        for distribution in distributions:
            for flip_y in label_noises:
                for spacing in ccs:
                    stream = StreamGenerator(
                        incremental=drift_type,
                        weights=distribution,
                        random_state=1994,
                        y_flip=flip_y,
                        concept_sigmoid_spacing=spacing,
                        n_drifts=n_drifts,
                        chunk_size=chunk_size,
                        n_chunks=2,
                        n_clusters_per_class=1,
                        n_features=8,
                        n_informative=8,
                        n_redundant=0,
                        n_repeated=0,
                    )
                    if spacing == None and drift_type == True:
                        pass
                    else:
                        streams.update({str(stream): stream})

    return streams
def river_hyper_plane_stream():
    # Step 1: Generate synthetic data using River
    n_samples = 1000  # Number of samples to generate

    # Example: Hyperplane generator from river
    dataset = synth.LEDDrift(seed=112, noise_percentage=0.28, irrelevant_features=True, n_drift_features=4)

    # Initialize lists to collect data and labels
    data = []
    labels = []

    # # Step 2: Generate the data
    # for _ in range(n_samples):
    #     x, y = dataset.ne
    #     data.append(list(x.values()))
    #     labels.append(y)
    for x, y in dataset.take(400000):
        data.append(x)
        labels.append(y)

    print(np.shape(data))
    print(np.unique(labels))
    print(len(data[0]))
    print(data[0].keys)
    n_features =len(data[0])
    arff_data = np.column_stack((data, labels))

    # Define the ARFF attributes
    attributes = [(f'feature_{i + 1}', 'REAL') for i in range(n_features)]
    attributes.append(('class', [str(i) for i in labels]))  # Assuming binary classification

    # Define ARFF dictionary
    arff_dict = {
        'description': 'Synthetic data generated using River',
        'relation': 'hyperplane_data',
        'attributes': attributes,
        'data': arff_data
    }

    # Step 4: Write the ARFF file manually
    with open('synthetic_data.arff', 'w') as f:
        # Write the ARFF header
        f.write(f"@RELATION hyperplane_data\n\n")
        for attr in attributes:
            if isinstance(attr[1], list):  # For nominal attributes (like 'class')
                attr_values = "{" + ",".join(attr[1]) + "}"
                f.write(f"@ATTRIBUTE {attr[0]} {attr_values}\n")
            else:
                f.write(f"@ATTRIBUTE {attr[0]} {attr[1]}\n")
        f.write("\n@DATA\n")

        # Write the data
        for row in arff_data:
            f.write(",".join(map(str, row)) + "\n")
def streams():
    # Variables
    # distributions = [[0.95, 0.05], [0.90, 0.10], [0.85, 0.15]]
    distributions = [[0.27, 0.23, 0.3, 0.2]]
    label_noises = [
        0.21,
    ]
    incremental = [True]
    ccs = [5]
    n_drifts = 50

    # Prepare streams
    streams = {}
    for drift_type in incremental:
        for distribution in distributions:
            for flip_y in label_noises:
                for spacing in ccs:
                    stream = StreamGenerator(
                        incremental=drift_type,
                        weights=distribution,
                        random_state=123,
                        y_flip=flip_y,
                        concept_sigmoid_spacing=spacing,
                        n_drifts=n_drifts,
                        chunk_size=2000,
                        n_chunks=200,
                        n_clusters_per_class=1,
                        n_classes=4,
                        n_features=8,
                        n_informative=8,
                        n_redundant=0,
                        n_repeated=0,
                    )
                    if spacing == None and drift_type == True:
                        pass
                    else:
                        streams.update({str(stream): stream})

    return streams

# streaam2()
