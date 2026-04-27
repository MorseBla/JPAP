import os
import subprocess
import time
import sys
import signal

solution = "3"      # 1=FC_GPU, 3=LQR/Proposed, 7=DFS
power_sp = "5000"
control_period = "1"

logs_base = ""


if (solution == "1"):
    logs_base = "logs/FC/"
elif (solution == "3"):
    logs_base = "logs/LQR/"
else:
    logs_base = "logs/DFS/"

if __name__ == '__main__':
    process_list = []

    if int(sys.argv[1]) == 4:
        t1 = sys.argv[2]
        t2 = sys.argv[3]
        t3 = sys.argv[4]
        t4 = sys.argv[5]
        

        os.makedirs(logs_base, exist_ok=True)
        
        with open(logs_base + "task_values.txt", 'w') as file:
            file.write(f"{t1}\n")
            file.write(f"{t2}\n")
            file.write(f"{t3}\n")


        command4 = [
            "./server",
            "3",
            solution,
            power_sp,
            control_period,
            str(sys.argv[7]),
            str(sys.argv[9]),
            str(sys.argv[11])
        ]

        # Start server first
        server_proc = subprocess.Popen(command4)
        pid4 = server_proc.pid
        process_list.append(server_proc)

        # Give server time to create shared memory
        time.sleep(0.5)

        command1 = ["./" + str(t1), "0", str(sys.argv[6]),  str(sys.argv[7]),  str(sys.argv[14]), logs_base + "task1", str(pid4)]
        command2 = ["./" + str(t2), "1", str(sys.argv[8]),  str(sys.argv[9]),  str(sys.argv[14]), logs_base + "task2", str(pid4)]
        command3 = ["./" + str(t3), "2", str(sys.argv[10]), str(sys.argv[11]), str(sys.argv[14]), logs_base + "task3", str(pid4)]

        p1 = subprocess.Popen(command1)
        p2 = subprocess.Popen(command2)
        p3 = subprocess.Popen(command3)

        process_list.extend([p1, p2, p3])

        print(command4)
        print(command1)
        print(command2)
        print(command3)

        start = time.time()
        timeout = float(sys.argv[14])
        switch = 0

        while process_list:
            for process in process_list[:]:
                if process.poll() is not None:
                    process_list.remove(process)
                    continue

                elapsed_time = time.time() - start

                if elapsed_time >= 0.5 * timeout and switch == 0:
                    print('\n *************** \n')
                    print('new task change\n')
                    print('*************** \n')

                    with open(logs_base + "task_values.txt", 'w') as file:
                        file.write(f"{t1}\n")
                        file.write(f"{t2}\n")
                        file.write(f"{t3}\n")
                        file.write(f"{t4}\n")

                    with open(logs_base + "setpoints.txt", 'w') as file:
                        file.write(f"{sys.argv[7]}\n")
                        file.write(f"{sys.argv[9]}\n")
                        file.write(f"{sys.argv[11]}\n")
                        file.write(f"{sys.argv[13]}\n")
                    command5 = [
                        "./" + str(t4),
                        "3",
                        str(sys.argv[12]),
                        str(sys.argv[13]),
                        str(sys.argv[14]),
                        logs_base + "task4",
                        str(pid4)
                    ]

                    print(command5)
                    p4 = subprocess.Popen(command5)
                    process_list.append(p4)

                    #print("Sending SIGINT to server PID:", pid4)
        
                    os.kill(pid4, signal.SIGUSR1)
                    #os.kill(pid4, signal.SIGINT)
                    switch = 1

                if elapsed_time >= timeout:
                    process.kill()
                    process_list.remove(process)

            time.sleep(0.1)
