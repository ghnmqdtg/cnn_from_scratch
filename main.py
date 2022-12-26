from utils.dataset_generator import DatasetGenerator
import config


if __name__ == '__main__':
    dataset_generator = DatasetGenerator(config.SRC_FOLDER)
    dataset = dataset_generator.prepare()
