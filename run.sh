#!/bin/bash

#python server.py & 
sleep 2 # Sleep for 2s to give the server enough time to start
python client.py --partition=0 &> output0.log & 
python client.py --partition=1 &> output1.log &
python client.py --partition=2 &> output2.log &
python client.py --partition=3 &> output3.log &
python client.py --partition=4 &> output4.log &
python client.py --partition=5 &> output5.log &
python client.py --partition=6 &> output6.log &
python client.py --partition=7 &> output7.log &
#python client.py --partition=8 &
#python client.py --partition=9 &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
sleep 86400
