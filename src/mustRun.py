from app import App
from model_factory import model_factory
from processing import start_processing
from train import start_training


def train_and_run():
    print("starting processing")
    start_processing()
    print("processing complete")
    print("creating model factory")
    mf = model_factory()
    print("creating model")
    model = mf.create_model()
    print("starting training")
    start_training(model)

    print("starting app")
    app = App()
    app.start()


def run_app():
    app = App()
    app.start()


def train():
    print("creating model factory")
    mf = model_factory()
    print("creating model")
    model = mf.create_model()
    print("starting training")
    start_training(model)


if __name__ == "__main__":
    # Uncomment the line below to run the full pipeline (processing + training + app)
    # train_and_run()

    # If you only want to run the app (after processing and training are done), use this:
    run_app()

    # If you only want to train the model (after processing is done), use this:
    # train()
