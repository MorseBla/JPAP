#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <thread>
#include<unistd.h>
#include <fstream>
#include<filesystem>
#include<sys/wait.h>
#include <cstdlib>
using namespace std;


struct Task {
    string name;
    float param1;
    float param2;
};



void filewrite(pid_t server_pid)
{
    
  ofstream outFile("pid.txt", std::ios::app);
  if (outFile.is_open()){
    outFile<<server_pid<<"\n";
      
  }      
  outFile.close();
}
void filedelete() {
    std::string path = "pid.txt";
    try {
        if (std::filesystem::remove(path)) {
            #ifdef DEBUG 
                std::cout << "File " << path << " deleted successfully.\n";
            #endif
        } else {
            #ifdef DEBUG 
                std::cout << "File " << path << " not found (nothing to delete).\n";
            #endif
        }
    } 
    catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
}
void launch_tasks(int num_tasks, unordered_map<int, Task>& taskset, float duration, string path, pid_t server_pid,
                 vector<int>& task_pids)
{
    for (int i = 0; i < num_tasks; i++) {
        pid_t pid = fork();

        if (pid < 0) {
            perror("Fork failed");
            return;
        }

        if (pid == 0) {
            // --- CHILD PROCESS ---
            string exec_path = taskset[i].name;
            //string exec_path = "../application/bin" + taskset[i].name;

            vector<string> arg_strings;
            arg_strings.push_back(exec_path);            // argv[0] is the program name
            arg_strings.push_back(to_string(i));     // taskno (1-indexed usually)
            arg_strings.push_back(to_string(taskset[i].param1));
            arg_strings.push_back(to_string(taskset[i].param2));
            arg_strings.push_back(to_string(duration));
            arg_strings.push_back(path);
            arg_strings.push_back(to_string(server_pid));
            vector<char*> args;
            for (auto& s : arg_strings) {
                args.push_back((char*)s.c_str());
            }
            args.push_back(nullptr);
            execvp(args[0], args.data());
            cout << exec_path << " AAAAAAAAAAAAAA\n";
            perror("execvp failed in scheduler");
            exit(1); 
        } else {
            filewrite(pid); 
            task_pids.push_back(pid);
        }
    }
}

void launch_server(int num_tasks, int solution, float duration,string path,float powersetpoint,
                   unordered_map<int, Task>& taskset, vector<int>& task_pids) 
{
    pid_t pid = fork();

    if (pid < 0) {
        perror("Fork failed for server");
        return;
    }
    task_pids.push_back(pid);
    if (pid == 0) {
        vector<string> arg_strings;
        arg_strings.push_back("../scheduler/bin/server"); 
        //arg_strings.push_back("./server");
        arg_strings.push_back(to_string(num_tasks));
        arg_strings.push_back(to_string(solution));
        arg_strings.push_back(to_string((int)duration));
        arg_strings.push_back(to_string((float)powersetpoint));
        arg_strings.push_back(path);
        for (int i = 0; i < num_tasks; i++) {
            arg_strings.push_back(to_string(taskset[i].param1)); // initial_rate
            arg_strings.push_back(to_string(taskset[i].param2)); // setpoint
        }
        vector<char*> args;
        for (auto& s : arg_strings) {
            args.push_back((char*)s.c_str());
        }
        args.push_back(nullptr);

        execvp(args[0], args.data());
        
        perror("Server execvp failed");
        exit(1);
    } 
}

int main(int argc, char*argv[]){
    
    int num_tasks = atoi(argv[1]);
    string path =argv[2];
    int solution = atoi(argv[3]);
    int duration = atoi(argv[4]);
    float powersetpoint = atof(argv[5]);
    int expected_len = 6 + 3 * num_tasks;
    unordered_map<int,Task>taskset;
    filedelete();
      /*
    argument i server
     int numtasks= atoi(argv[1]);
    int solution = std::atoi(argv[2]);
    int duration = std::atoi(argv[3]);
    string path= argv[4];
    */
    int base = 6; 
    for (int i = 0; i < num_tasks; i++) {
        if (base + 2 < argc) {
            Task t;
            t.name = argv[base];
            t.param1 = atof(argv[base + 1]);
            t.param2 = atof(argv[base + 2]);
            
            taskset[i] = t;
            base += 3;
        }
    }
#ifdef DEBUG 
    cout<<"Power Set point = "<<powersetpoint<<endl;
    cout << "Tasks loaded:" << endl;
    for (auto const& [id, task] : taskset) {
        cout << "ID: " << id << " | Name: " << task.name 
             << " | period: " << task.param1 << " | setpoint: " << task.param2 << endl;
    }
#endif

    vector<int> task_pids;
    vector<int> procs;
    launch_server(num_tasks,solution,duration,path,powersetpoint,taskset,task_pids); 
    pid_t server_pid = task_pids.back();
    task_pids.pop_back();                

    std::this_thread::sleep_for(std::chrono::milliseconds(5000)); 
    launch_tasks(num_tasks,taskset,duration,path,server_pid,task_pids);
    int status;
    waitpid(server_pid, &status, 0); 
    std::cout<<"Server released Scheduling \n";
    for (pid_t pid : task_pids) {
       waitpid(pid, &status, 0);
        
    }   
#ifdef DEBUG 
    std::cout<<"Finished Scheduling \n";
#endif
    return 0;
}
