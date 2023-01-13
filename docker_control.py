import docker
import os
import logging

# Debug
logging.basicConfig(
    format='%(asctime)s | %(levelname)s: %(message)s', level=logging.INFO)

# Get the current working directory
logging.info("Looking for the current directory...")
cwd = os.getcwd()
logging.info(f"The current directory is --> {cwd}")

logging.info("Looking for the current display...")
display = os.getenv("DISPLAY")
logging.info(f"The current display is --> {display}")


# Docker SDK --> https://docker-py.readthedocs.io/en/stable/containers.html

logging.info("Instantiating Docker client...")
client = docker.from_env()

logging.info("Checking if the Image already exists...")

if not os.system('[ "$(docker image inspect yolov7_detect_track_count:latest 2> /dev/null)" == [] ]'):

    logging.info("The image doesn't exist. Building the Docker Image...")

    os.system("cd ./dependencies")
    os.system("docker build -t yolov7_detect_track_count .")
    os.system("cd ..")

else:
    logging.info("The image has already exists.")


logging.info(
    "Setting up X Server to accept connections. Turning Access control off...")
os.system("xhost +")


logging.info("Running a container...")
container = client.containers.run(
    image='yolov7_detect_track_count:latest',
    auto_remove=True,
    device_requests=[docker.types.DeviceRequest(
        device_ids=["0"], capabilities=[['gpu']])],
    devices=["/dev/video0:/dev/video0"],
    volumes={
        '/tmp/.X11-unix': {'bind': '/tmp/.X11-unix', 'mode': 'rw'},
        cwd: {'bind': '/workspace', 'mode': 'rw'}
    },
    environment=[f"DISPLAY={display}"],
    tty=True,
    command='python test_oop.py'
)
