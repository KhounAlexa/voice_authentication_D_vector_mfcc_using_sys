############################## Dependencies Install ######################################
# pip install audiomentations
# pip install pyrubberband
# pip install librosa
# pip install numpy
# pip install pandas
# pip install h5py
# pip install tqdm
# pip install torch torchvision torchaudio
# pip install matplotlib
# pip install openpyxl

from constant.Constant import *


######################### Data Preparation #########################

# List All in Directory
def list_all(directory: str, ignore_file_names=None) -> list[str]:
    if ignore_file_names is None:
        ignore_file_names = ['.ipynb_checkpoints', '.DS_Store']
    filenames = [file_name for file_name in os.listdir(directory) if file_name not in ignore_file_names]

    return filenames


# Pre-processing
if len(sys.argv) < 2:
    print('Usage: python train_and_authenticate.py <user_name>')
    exit(-1)
user_directory = sys.argv[1]
OWNER_ORIGINAL = f'./original/source/owner/{user_directory}/'
OTHER_ORIGINAL = './original/source/other/'


def collect_original_data():
    owner_filenames = list_all(OWNER_ORIGINAL)
    other_filenames = list_all(OTHER_ORIGINAL)

    owner_dataframe = DataFrame(owner_filenames, columns=['filename'])
    owner_dataframe['label'] = 0

    other_dataframe = DataFrame(other_filenames, columns=['filename'])
    other_dataframe['label'] = 1

    def rename_filepath(filename: str, owner: bool) -> str:
        if owner:
            return f'./original/source/owner/{user_directory}/' + filename
        else:
            return f'./original/source/other/' + filename

    owner_dataframe['file_rename'] = owner_dataframe['filename'].apply(lambda x: rename_filepath(x, True))
    other_dataframe['file_rename'] = other_dataframe['filename'].apply(lambda x: rename_filepath(x, False))

    owner_dataframe = owner_dataframe[['file_rename', 'label']]
    owner_dataframe.columns = ['filename', 'label']
    other_dataframe = other_dataframe[['file_rename', 'label']]
    other_dataframe.columns = ['filename', 'label']

    # print(owner_dataframe)
    # print(other_dataframe)

    owner_dataframe.to_csv(f'./labels/owner_{user_directory}.csv', index=False)
    other_dataframe.to_csv('./labels/other.csv', index=False)
    pass


def split_original_data():
    start_time = time.time()  # Start timer
    owner_dataframe = pandas.read_csv(f'./labels/owner_{user_directory}.csv')
    other_dataframe = pandas.read_csv('./labels/other.csv')

    owner_dataframe = owner_dataframe.sample(frac=FRAC, random_state=168).reset_index(drop=True)
    other_dataframe = other_dataframe.sample(frac=0.2, random_state=168).reset_index(drop=True)
    print((owner_dataframe['filename'].apply(lambda x: x.split('/')[-1])))
    print((other_dataframe['filename'].apply(lambda x: x.split('/')[-1])))
    train_owner_dataframe = owner_dataframe[:int(len(owner_dataframe) * 0.9)]
    test_owner_dataframe = owner_dataframe[int(len(owner_dataframe) * 0.9):]
    print(len(train_owner_dataframe))
    print(len(test_owner_dataframe))

    train_other_dataframe = other_dataframe[:int(len(other_dataframe) * 0.9)]
    test_other_dataframe = other_dataframe[int(len(other_dataframe) * 0.9):]
    # print(len(train_other_dataframe))
    # print(len(test_other_dataframe))

    # Merge Train Data
    train_dataframe = pandas.concat([train_owner_dataframe, train_other_dataframe], ignore_index=True)
    train_dataframe = train_dataframe.sample(frac=1, random_state=168).reset_index(drop=True)
    train_dataframe.to_csv('./labels/train.csv', index=False)

    # Merge Test Data
    test_dataframe = pandas.concat([test_owner_dataframe, test_other_dataframe], ignore_index=True)
    test_dataframe = test_dataframe.sample(frac=1, random_state=168).reset_index(drop=True)
    test_dataframe.to_csv('./labels/test.csv', index=False)

    # print((train_dataframe))
    # print((test_dataframe))

    pass


SPLIT_SECONDS = 5


def segment_audio_files():
    def segment_audio_file(filename: str, label: int) -> DataFrame:
        # Load audio file
        y, sr = librosa.load(filename, sr=None)

        # Convert to mono if not already
        if y.ndim > 1:
            y = librosa.to_mono(y)

        # Resample to 16 kHz
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000

        # Convert to 16-bit
        y = (y * 32767).astype('int16')

        temp_dataframe = DataFrame()
        segment_samples = SPLIT_SECONDS * sr
        total_segments = len(y) // segment_samples
        for i in range(total_segments + 1):
            start = i * segment_samples
            end = start + segment_samples if (start + segment_samples) < len(y) else len(y)
            segment = y[start:end]

            # Skip segments shorter than 2 seconds
            if len(segment) < 2 * sr:
                continue

            # Save as 16-bit mono 16 kHz
            segment_filename = f"{filename.replace(f'source/owner/{user_directory}', 'preprocessed').replace('source/other', 'preprocessed').replace('.wav', f'_{i}.wav')}"
            sf.write(os.path.join(segment_filename), segment, sr, subtype='PCM_16')

            # Add segment_file with label to DataFrame
            temp_dataframe = pandas.concat(
                [temp_dataframe, pandas.DataFrame([[segment_filename, label]], columns=['filename', 'label'])],
                ignore_index=True)

        return temp_dataframe

    original_train_dataframe = pandas.read_csv('./labels/train.csv')
    original_test_dataframe = pandas.read_csv('./labels/test.csv')

    train_dataframe = DataFrame()
    test_dataframe = DataFrame()

    for index, row in original_train_dataframe.iterrows():
        train_dataframe = pandas.concat([train_dataframe, segment_audio_file(row['filename'], row['label'])],
                                        ignore_index=True)

    for index, row in original_test_dataframe.iterrows():
        test_dataframe = pandas.concat([test_dataframe, segment_audio_file(row['filename'], row['label'])],
                                       ignore_index=True)

    print("TRAIN:", len(train_dataframe), "FILES.")
    print("TEST:", len(test_dataframe), "FILES.")
    train_dataframe.to_csv('./labels/train_labels.csv', index=False)
    test_dataframe.to_csv('./labels/test_labels.csv', index=False)
    pass


######################### Feature Extraction Function #########################
def extract_mfcc(file_path, n_mfcc=40):
    """
    Extracts MFCC features from an audio file.
    """
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    # Normalize MFCC
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
    return mfcc.T  # Shape: (Time, n_mfcc)


#########################  Collate Function #########################
def collate_fn(batch):
    """Custom collate function to pad variable-length sequences."""
    inputs, labels = zip(*batch)
    # Pad the input sequences
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    # Stack the labels
    labels = torch.stack(labels)
    return inputs_padded, labels


######################### Define the Model #########################
class DVectorModel(nn.Module):
    """
    Neural network model to extract D-vectors (embeddings) from MFCC features.
    """

    def __init__(self, input_dim=40, embedding_dim=128):
        super(DVectorModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(256)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

        self.conv2 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(256)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

        self.conv3 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Linear(256, embedding_dim)

    def forward(self, x):
        # x: (batch, Time, n_mfcc)
        x = x.transpose(1, 2)  # (batch, n_mfcc, Time)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)  # (batch, 256, 1)
        x = x.squeeze(2)  # (batch, 256)
        embeddings = self.fc(x)  # (batch, embedding_dim)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


######################### DataLoader #########################
class VoiceDataset(Dataset):
    """
    PyTorch Dataset for voice data using a DataFrame and optional augmentations.
    """

    def __init__(self, dataframe, n_mfcc=40, augment=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame with 'filename' and 'label' columns.
            n_mfcc (int): Number of MFCC features.
            augment (callable, optional): Augmentation function to apply to audio data.
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.n_mfcc = n_mfcc
        self.augment = augment

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get the filename and label from the DataFrame
        filename = self.dataframe.loc[idx, 'filename']
        label = self.dataframe.loc[idx, 'label']

        # Load audio
        y, sr = librosa.load(filename, sr=None)

        # Apply augmentation if provided
        if self.augment:
            y = self.augment(samples=y, sample_rate=sr)

        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        # Normalize MFCC
        mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
        # Convert to tensor
        mfcc = torch.tensor(mfcc.T, dtype=torch.float)  # Shape: (Time, n_mfcc)
        label = torch.tensor(label, dtype=torch.long)
        return mfcc, label


######################### Custom Augmentation Classes #########################
class SpeedPerturbation(BaseWaveformTransform):
    supports_multichannel = False

    def __init__(self, min_speed=0.8, max_speed=1.2, p=0.5):
        super().__init__(p)
        self.min_speed = min_speed
        self.max_speed = max_speed

    def apply(self, samples, sample_rate, **params):
        speed_rate = numpy.random.uniform(self.min_speed, self.max_speed)
        return pyrubberband.time_stretch(y=samples, sr=sample_rate, rate=speed_rate)


######################### FrequencyMasking #########################
class FrequencyMasking(BaseWaveformTransform):
    supports_multichannel = False

    def __init__(self, min_mask_fraction=0.0, max_mask_fraction=0.15, num_masks=2, p=0.5):
        super().__init__(p)
        self.min_mask_fraction = min_mask_fraction
        self.max_mask_fraction = max_mask_fraction
        self.num_masks = num_masks

    def apply(self, samples, sample_rate, **params):
        stft = librosa.stft(samples)
        magnitude, phase = numpy.abs(stft), numpy.angle(stft)

        masked_magnitude = self.apply_frequency_masking(magnitude)
        masked_stft = masked_magnitude * numpy.exp(1j * phase)
        return librosa.istft(masked_stft)

    def apply_frequency_masking(self, magnitude_spectrogram):
        masked_magnitude = magnitude_spectrogram.copy()
        num_freq_bins = masked_magnitude.shape[0]

        for _ in range(self.num_masks):
            mask_fraction = numpy.random.uniform(self.min_mask_fraction, self.max_mask_fraction)
            mask_width = int(mask_fraction * num_freq_bins)
            if mask_width == 0:
                continue
            mask_start = numpy.random.randint(0, num_freq_bins - mask_width)
            masked_magnitude[mask_start:mask_start + mask_width, :] = 0

        return masked_magnitude


######################### BackgroundNoiseMixing #########################
class BackgroundNoiseMixing(BaseWaveformTransform):
    supports_multichannel = False

    def __init__(self, path_to_sound='noise', min_snr_in_db=0, max_snr_in_db=15, p=0.5):
        super().__init__(p)
        self.path_to_sound = path_to_sound
        self.min_snr_in_db = min_snr_in_db
        self.max_snr_in_db = max_snr_in_db
        self.noise_file_paths = self._load_noise_file_paths()

    def _load_noise_file_paths(self):
        noise_files = [
            os.path.join(self.path_to_sound, fname)
            for fname in os.listdir(self.path_to_sound)
            if fname.lower().endswith('.wav')
        ]
        if not noise_files:
            raise ValueError(f"No noise files found in {self.path_to_sound}")
        return noise_files

    def apply(self, samples, sample_rate, **params):
        noise_file_path = random.choice(self.noise_file_paths)
        noise_samples, _ = librosa.load(noise_file_path, sr=sample_rate)
        noise_samples = self._match_length(noise_samples, len(samples))

        snr_db = numpy.random.uniform(self.min_snr_in_db, self.max_snr_in_db)
        adjusted_noise = self._adjust_noise_level(samples, noise_samples, snr_db)

        augmented_samples = samples + adjusted_noise
        max_amplitude = numpy.max(numpy.abs(augmented_samples))
        if max_amplitude > 1.0:
            augmented_samples = augmented_samples / max_amplitude

        return augmented_samples

    @staticmethod
    def _match_length(noise_samples, target_length):
        if len(noise_samples) < target_length:
            repeats = int(numpy.ceil(target_length / len(noise_samples)))
            noise_samples = numpy.tile(noise_samples, repeats)
        return noise_samples[:target_length]

    @staticmethod
    def _adjust_noise_level(signal, noise, snr_db):
        rms_signal = numpy.sqrt(numpy.mean(signal ** 2))
        rms_noise = numpy.sqrt(numpy.mean(noise ** 2))
        snr_linear = 10 ** (snr_db / 20)
        rms_noise_adjusted = rms_signal / snr_linear
        adjustment_factor = rms_noise_adjusted / rms_noise if rms_noise != 0 else 0
        return noise * adjustment_factor


######################### Training Function #########################
def train_model(model, train_loader, optimizer, num_epochs, device):
    """
    Trains the D-vector model using Triplet Loss.
    """
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2).to(device)
    scaler = GradScaler()
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with autocast():
                # Separate owner and other samples
                owner_indices = (labels == 0).nonzero(as_tuple=True)[0]
                other_indices = (labels == 1).nonzero(as_tuple=True)[0]

                if len(owner_indices) == 0 or len(other_indices) == 0:
                    # Skip if no samples in a class
                    continue

                # Create triplets
                # Randomly pair owner (anchor & positive) and other (negative)
                anchor = inputs[owner_indices]
                positive = inputs[owner_indices]
                negative = inputs[other_indices]

                # Ensure the number of triplets is the same
                min_len = min(len(anchor), len(negative))
                anchor = anchor[:min_len]
                positive = positive[:min_len]
                negative = negative[:min_len]

                if len(anchor) == 0:
                    continue

                # Forward pass
                anchor_emb = model(anchor)  # (batch, embedding_dim)
                positive_emb = model(positive)  # (batch, embedding_dim)
                negative_emb = model(negative)  # (batch, embedding_dim)

                # Compute Triplet Loss
                loss = triplet_loss(anchor_emb, positive_emb, negative_emb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    print("Training completed.")


######################### Saving and Loading the Model #########################
def save_model_h5(model, path):
    """
    Saves the model's state_dict to an .h5 file.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save the model's state dict to .h5 file
    with h5py.File(path, 'w') as f:
        for key, value in model.state_dict().items():
            value_data = value.cpu().numpy()
            f.create_dataset(key, data=value_data)
    print(f"Model successfully saved to {path}")


def load_model_h5(path, device, input_dim, embedding_dim):
    """
    Loads the model's state_dict from an .h5 file.
    """
    # Initialize the model
    model = DVectorModel(input_dim=input_dim, embedding_dim=embedding_dim)

    # Load the model's state dict from .h5 file
    with h5py.File(path, 'r') as f:
        state_dict = {}
        for key in f.keys():
            data = torch.tensor(f[key][()])
            state_dict[key] = data
    model.load_state_dict(state_dict)

    # Move model to the specified device (CPU/GPU)
    model.to(device)
    model.eval()
    print(f"Model successfully loaded from {path}")
    return model

######################### Compute Embeddings and Centroid #########################

def get_embedding(filename, model, device):
    """
    Extracts the embedding for a single audio file.
    """
    model.to(device)
    model.eval()
    with torch.no_grad():
        mfcc = extract_mfcc(filename, N_MFCC)
        mfcc_tensor = torch.tensor(mfcc, dtype=torch.float).unsqueeze(0).to(device)  # (1, Time, n_mfcc)
        embedding = model(mfcc_tensor)  # (1, embedding_dim)
        return embedding.squeeze(0).cpu().numpy()


def compute_centroid(embeddings):
    """
    Computes the centroid of embeddings.
    """
    centroid = np.mean(embeddings, axis=0)
    centroid /= np.linalg.norm(centroid)  # Normalize
    return centroid


######################## Main Training Function #######################


def main_training():
    # Load the training DataFrame
    train_df = pd.read_csv(TRAIN_CSV)
    train_df = pd.concat([train_df] * LOAD, ignore_index=True)
    train_df = train_df.sample(frac=1, random_state=168).reset_index(drop=True)
    print(f"Number of training samples before filtering: {len(train_df)}")

    # Define the augmentation pipeline
    augment = Compose([
        SpeedPerturbation(p=0.5),
        # PitchShifting(p=0.5),
        Gain(min_gain_db=-12, max_gain_db=12, p=0.5),
        TimeMask(min_band_part=0.0, max_band_part=0.05, p=0.5),
        FrequencyMasking(p=0.5),
        BackgroundNoiseMixing(path_to_sound='noise', p=0.5),
    ])

    # Create the training Dataset and DataLoader with augmentation
    print("\nPreparing training dataset and dataloader...")
    train_dataset = VoiceDataset(train_df, n_mfcc=N_MFCC, augment=augment)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=0,  # Adjust based on your CPU
        pin_memory=True
    )
    print(f"Number of training samples: {len(train_dataset)}")

    # Initialize the model and optimizer
    print("\nInitializing model and optimizer...")
    model = DVectorModel(input_dim=INPUT_DIM, embedding_dim=EMBEDDING_DIM)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Train the model
    print("\nStarting training...")
    train_model(model, train_loader, optimizer, NUM_EPOCHS, DEVICE)

    # Save the trained model
    MODEL_PATH = f'./models/voice_auth_{user_directory}_model.h5'
    save_model_h5(model, MODEL_PATH)

    # Compute the owner centroid using the owner's training data (without augmentation)
    print("\nComputing owner centroid...")
    owner_train_df = train_df[train_df['label'] == 0].reset_index(drop=True)
    owner_embeddings = []
    for idx in tqdm(range(len(owner_train_df)), desc="Computing owner embeddings"):
        filename = owner_train_df.loc[idx, 'filename']
        emb = get_embedding(filename, model, DEVICE)
        owner_embeddings.append(emb)
    owner_centroid = compute_centroid(owner_embeddings)
    print("Owner centroid computed.")

    # Save the owner centroid
    centroid_path = f'./models/owner_{user_directory}_centroid.h5'
    with h5py.File(centroid_path, 'w') as f:
        f.create_dataset('centroid', data=np.array(owner_centroid))

    print(f"Owner centroid saved to {centroid_path}.")
    print("Model is ready for authentication.")


################# Authentication Functions ################################

def load_owner_centroid(centroid_path):
    """
    Loads the owner's centroid from an .h5 file.
    """
    with h5py.File(centroid_path, 'r') as f:
        owner_centroid = f['centroid'][:]
    return owner_centroid


def authenticate(filename, owner_centroid, model, device, threshold):
    """
    Authenticates a single audio file by comparing its embedding to the owner's centroid.
    Returns:
        match (bool): True if similarity >= threshold, else False
        similarity (float): Cosine similarity score
    """
    emb = get_embedding(filename, model, device)
    emb /= np.linalg.norm(emb)
    similarity = cosine_similarity([emb], [owner_centroid])[0][0]
    return similarity >= threshold, similarity


def authenticate_samples(test_df, owner_centroid, model, device, threshold):
    """
    Authenticates samples from a DataFrame.
    Returns a list of results for each file: [(filename, isOwner, pred, similarity)].
    """
    results = []
    for idx in tqdm(range(len(test_df)), desc="Authenticating samples"):
        filename = test_df.loc[idx, 'filename']
        is_owner = test_df.loc[idx, 'label'] == 0
        match, similarity = authenticate(filename, owner_centroid, model, device, threshold)
        print(f"File: {os.path.basename(filename)}, Match: {match}, Similarity: {similarity:.4f}")
        results.append((filename, is_owner, match, similarity))
    return results


####################### Processing and Saving Results #################################
def process_and_save_data(results, output_file):
    # Convert results to a DataFrame
    df = pd.DataFrame(results, columns=['filename', 'isOwner', 'pred', 'similarity'])
    # Save the DataFrame to an Excel file
    df.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")


#################### Evaluation Function #########################
def evaluate_results(file_path):
    #print file_path
    print(file_path)
    # Load the Excel file into a DataFrame
    df = pd.read_excel(file_path, usecols=['filename', 'isOwner', 'pred'])

    # Convert 'isOwner' and 'pred' columns to boolean if they aren't already
    df['isOwner'] = df['isOwner'].astype(bool)
    df['pred'] = df['pred'].astype(bool)

    # Calculate confusion matrix
    y_true = df['isOwner']  # Actual values
    y_pred = df['pred']  # Predicted values

    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    # Calculate metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Display results
    print(f'\nConfusion Matrix:\n{cm}')
    print(f'\nAccuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall (Sensitivity): {recall:.2f}')
    print(f'Specificity: {specificity:.2f}')
    print(f'F1 Score: {f1_score:.2f}')

    # Plot confusion matrix
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Other', 'Owner'])
    cm_display.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()
    # return all
    return accuracy, precision, recall, specificity, f1_score


######################### Main Authentication Function #####################

def main_authentication():
    # Paths
    MODEL_PATH = f'./models/voice_auth_{user_directory}_model.h5'
    CENTROID_PATH = f'./models/owner_{user_directory}_centroid.h5'
    MAX_ATTEMPTS = 3  # Maximum number of retraining attempts

    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"\nAttempt {attempt} for authentication process:")

        # Load the trained model
        print("\nLoading the trained model for authentication...")
        model_loaded = load_model_h5(MODEL_PATH, DEVICE, input_dim=INPUT_DIM, embedding_dim=EMBEDDING_DIM)
        print("Model is ready for authentication.")

        # Load the owner's centroid
        owner_centroid = load_owner_centroid(CENTROID_PATH)
        print(f"Owner centroid loaded from {CENTROID_PATH}.")

        # Load the test DataFrame
        test_df = pd.read_csv(TEST_CSV)

        # Create the test Dataset and DataLoader without augmentation
        print("\nPreparing test dataset and dataloader...")
        test_dataset = VoiceDataset(test_df, n_mfcc=N_MFCC, augment=None)
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True
        )

        # Authenticate test samples
        print("\nAuthenticating test samples...")
        results = []
        for idx in tqdm(range(len(test_df)), desc="Authenticating samples"):
            filename = test_df.loc[idx, 'filename']
            is_owner = test_df.loc[idx, 'label'] == 0
            match, similarity = authenticate(filename, owner_centroid, model_loaded, DEVICE, threshold=THRESHOLD)
            print(f"File: {os.path.basename(filename)}, Match: {match}, Similarity: {similarity:.4f}")
            results.append((filename, is_owner, match, similarity))

        # Process and save results
        output_file = f'./results/authentication_results_with_{NUM_EPOCHS}_epochs_batch_size_{BATCH_SIZE}_with_frac_{FRAC}_lr_{LEARNING_RATE}_and_load_{LOAD}_user_{USER}.xlsx'
        process_and_save_data(results, output_file)

        # Evaluate results
        accuracy, precision, recall, specificity, f1_score = evaluate_results(output_file)

        print(f"\nAttempt {attempt} Evaluation Metrics:")
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Specificity: {specificity:.4f}, F1 Score: {f1_score:.4f}")

        # Check if the model meets the desired criteria
        if recall == 1.0 and accuracy > 0.65:
            print("\nRecall is 1.0 and accuracy is > 0.65. No further training needed.")
            break

        # If criteria not met, retrain the model
        if attempt < MAX_ATTEMPTS:
            print("\nCriteria not met. Retraining the model...")
            main_training()
        else:
            print("\nMaximum attempts reached. Taking the last result.")

    # Print the final evaluation metrics
    print(f"\nFinal Evaluation Metrics after {attempt} attempts:")
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Specificity: {specificity:.4f}, F1 Score: {f1_score:.4f}")

# Ensure the function `main_training` is correctly implemented to retrain the model as needed.


if __name__ == "__main__":

    start_time = time.time()  # Start timer
    collect_original_data()
    split_original_data()
    segment_audio_files()
    end_time = time.time()  # End timer
    print(f"Execution time for `Preprocessed`: {end_time - start_time:.2f} seconds")

    # Train the model
    if not os.path.exists(f'./models/voice_auth_{user_directory}_model.h5'):
        start_time = time.time()  # Start timer
        main_training()
        end_time = time.time()  # End timer
        minute = (end_time - start_time) / 60
        print(f"Execution time for `Training is `: {minute:.2f} minutes")

    # Authenticate and evaluate
    start_time = time.time()  # Start timer
    main_authentication()
    end_time = time.time()  # End timer
    print(f"Execution time for `Authenticate is `: {end_time - start_time:.2f} seconds")
