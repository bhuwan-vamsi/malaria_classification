# Malaria Parasite Species Classification Project

This project applies machine learning to classify malaria parasite species based on microscopic images, using a deep learning approach with the VGG16 architecture. Custom layers and image augmentation enhance the model’s performance and robustness.

## Dataset
The dataset, available on Kaggle, is provided by Loddo et al. through the Malaria Parasite Image Database (MP-IDB). It contains labeled images for different malaria species. For more information:

- **Kaggle Dataset:** [Malaria Parasite Species Dataset](https://www.kaggle.com/datasets/saife245/malaria-parasite-image-malaria-species)
- **Full Dataset Description:** [MP-IDB Description](https://link.springer.com/chapter/10.1007/978-3-030-13835-6_7)

## Citation
Loddo, A., Di Ruberto, C., Kocher, M., Prod’Hom, G. (2019). MP-IDB: The Malaria Parasite Image Database for Image Processing and Analysis. In: Lepore, N., Brieva, J., Romero, E., Racoceanu, D., Joskowicz, L. (eds) Processing and Analysis of Biomedical Information. SaMBa 2018. Lecture Notes in Computer Science(), vol 11379. Springer, Cham. https://doi.org/10.1007/978-3-030-13835-6_7

## Project Structure

- **ImageDataGenerator:** Uses Keras for data augmentation, applying transformations like rotation, brightness adjustments, and random flips to increase image diversity during training.
- **VGG16 Model:** Implements transfer learning with the VGG16 architecture and custom fully connected layers with dropout regularization.
- **Metrics:** Model performance is measured using accuracy, AUC, precision, and recall.

## Dependencies

- Python 3.7+
- TensorFlow (2.x)
- Keras
- Matplotlib
- Numpy
- JSON

## Code Overview

### Key Functions

- **`load_existing_results`**: Loads previously saved training results from a JSON file.
- **`save_results`**: Saves new metrics and hyperparameters to JSON for recording each training instance.
- **`get_data_generators`**: Creates data generators for training and validation using Keras' `ImageDataGenerator` with multiple transformations.
- **`build_vgg16_model`**: Builds and compiles a modified VGG16 model with custom layers and an Adam optimizer.
- **`train_and_evaluate_model`**: Trains the model, evaluates performance, saves the model, calculates metrics, and saves the model based on hyperparameters.

### Main Code Block

The script initializes a set of hyperparameters, calls `train_and_evaluate_model`, prints metrics, and stores results.

## Usage

1. Set the dataset directory path in `dataset_dir`.
2. Configure any desired hyperparameters in `param_dict`.
3. Run the script to train, evaluate, and save the model.
4. Results are saved in JSON format for easy reference.

## Example

```python
# Initialize hyperparameters
param_dict = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'dropout_rate': 0.5,
    'activation_function': 'relu',
    'optimizer': 'adam'
}

# Train and evaluate the model
metrics = train_and_evaluate_model(**param_dict)
print(metrics)

# Save results
save_results(metrics)
