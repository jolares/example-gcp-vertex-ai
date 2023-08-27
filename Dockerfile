FROM gcr.io/deeplearning-platform-release/tf2-cpu.2-6

WORKDIR /

# Copies trainer code to the docker image.
COPY src/trainer /trainer

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "trainer.train.tensorflow"]