import os
import time
from signal import signal, SIGINT
from sys import exit
from multiprocessing import Process

from server import start_server
from client import start_client
from myutils import get_dataset

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# All created processes
processes = []

def sig_handler(signal, frame):
    print("SIGINT or CTRL-C detected.")
    for p in processes:
        p.terminate()
    exit(0)
    

def simulation(num_rounds: int, num_clients: int, min_fit_clients: int):
    """
    Start simulation.
    """

    # Run handler function when signal is recieved
    signal(SIGINT, sig_handler)

    # Start the server
    server_process = Process(
        target=start_server, args=(num_rounds, num_clients, min_fit_clients)
    )
    server_process.start()
    processes.append(server_process)

    # Block execution for 2 sec so that server has time to start
    time.sleep(2)

    # Load dataset
    clients, train_data, test_data = get_dataset("femnist")

    # Start clients
    cid=0
    for c in clients:
        client_process = Process(target=start_client, args=(c, train_data[c], test_data[c]))
        client_process.start()
        print("Started client {} of total {} clients".format(cid, len(clients)))
        processes.append(client_process)
        cid +=1
        if (cid == num_clients):
            break

    # Block until all processes are finished
    for p in processes:
        p.join()


if __name__ == "__main__":
    simulation(num_rounds=100, num_clients=10, min_fit_clients=3)