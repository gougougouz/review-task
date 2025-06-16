#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <sys/time.h>

static int count = 0; // 记录触发次数

// SIGALRM信号处理函数
void alarm_handler(int sig) {
    count++; // 每次信号触发时增加计数
}

int main() {
    struct sigaction sa;
    struct itimerval timer;
    
    // 设置信号处理函数
    sa.sa_handler = alarm_handler;
    sa.sa_flags = 0;
    sigemptyset(&sa.sa_mask);
    if (sigaction(SIGALRM, &sa, NULL) == -1) {
        perror("sigaction");
        exit(1);
    }
    
    // 设置定时器：初始延迟1秒，之后每隔1秒触发一次
    timer.it_value.tv_sec = 1;
    timer.it_value.tv_usec = 0;
    timer.it_interval.tv_sec = 1;
    timer.it_interval.tv_usec = 0;
    
    if (setitimer(ITIMER_REAL, &timer, NULL) == -1) {
        perror("setitimer");
        exit(1);
    }
    
    // 主循环
    while (1) {
        pause(); // 等待信号
        // 输出当前字母（A-Z循环）
        printf("%c", 'A' + (count % 26));
        fflush(stdout); // 立即刷新输出缓冲区
    }
    
    return 0;
}