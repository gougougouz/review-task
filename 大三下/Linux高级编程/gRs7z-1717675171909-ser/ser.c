#include <stdio.h>
#include <string.h>
#include <netinet/in.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/socket.h>
#include <pthread.h>
#define SERV_PORT 8888					
#define Len 19
char *s="One World One Dream";
void mysend(int connfd,int start,int len){
	int i;
	char *p=s+start;
	for(i=0;i<len;i++){
		if(send(connfd,p+i,1,0)<1){
			printf("send character %dth error\n",start+i+1);
			exit(1);
		}
		else
			printf("%dth character is sent.\n",start+i+1);
			sleep(1);
	}	
}
void *do_work(void *arg)
{
	int n, i;
	int connfd=(int)(long)arg;
	char buf[2];//buf[0]:start buf[1]:len
	pthread_detach(pthread_self());
	n=recv(connfd,buf,2,MSG_DONTWAIT);
	if(n==-1){
		if(errno==EAGAIN||errno==EWOULDBLOCK||errno==EINTR){
			sleep(1);
			n=recv(connfd,buf,2,MSG_DONTWAIT);
			if(n>0)
				goto l1;
		}
		goto end;
	}
	else if(n==0){
		printf("Client Socket closed.\n");
		goto end;
	}
l1:	
	if(buf[0]<0||buf[0]>=Len||buf[1]<1||buf[0]+buf[1]-1>=Len){
		printf("download parameters error\n");
		goto end;
	}
	mysend(connfd,buf[0],buf[1]);
end:		
	close(connfd);
		
}
int main(void)
{
	struct sockaddr_in servaddr;
	int listenfd, connfd,i=0;
	pthread_t tid;
	listenfd = socket(AF_INET, SOCK_STREAM, 0);
	if(listenfd==-1){
		perror("listen socket create error");
		exit(1);
	}
	bzero(&servaddr, sizeof(servaddr));
	servaddr.sin_family = AF_INET;
	servaddr.sin_addr.s_addr = htonl(INADDR_ANY);
	servaddr.sin_port = htons(SERV_PORT);
	int ret=bind(listenfd, (struct sockaddr *)&servaddr, sizeof(servaddr));
	if(ret==-1){
		perror("bind error");
		exit(1);
	}
	ret=listen(listenfd, 200);
	if(ret==-1){
		perror("listen function error");
		exit(1);
	}
	printf("Accepting connections ...\n");
	while (1) {
		connfd = accept(listenfd, NULL,NULL);
		if(connfd==-1){
			perror("accept function error");
			continue;
		}	
		pthread_create(&tid, NULL, do_work, (void*)(long)connfd);
	}
	close(connfd);
	return 0;
}
