#!/bin/bash

#python server.py & 
sleep 2 # Sleep for 2s to give the server enough time to start

for number in {0..50}
do
python client.py --cid=$number &> ./output/output$number.log &
done

#python client.py --cid=0 &> output0.log & 
#python client.py --cid=1 &> output1.log &
#python client.py --cid=2 &> output2.log &
#python client.py --cid=3 &> output3.log &
#python client.py --cid=4 &> output4.log &
#python client.py --cid=5 &> output5.log &
#python client.py --cid=6 &> output6.log &
#python client.py --cid=7 &> output7.log &
#python client.py --cid=8 &> output8.log &
#python client.py --cid=9 &> output9.log &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
sleep 86400
