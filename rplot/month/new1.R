setwd('D:/R')
#导入包
library('fpp3')
library('ggfortify')
library('ggplot2')
library('Rmisc')
#筛选数据，处理日期格式
raw_data=read.csv("data_2015_2021.csv",encoding="UTF-8")
data=raw_data[,-1]
data1<-data %>% filter(.,data$type %in% c('PM2.5_24h','PM10_24h','SO2_24h','SO2_24h','O3_24h','AQI'),data$hour==23)
data1=data1[,-2]
pdata<- data1 %>%mutate(date = ymd(date))
PM2.5_data<-pdata%>%filter(pdata$type=='PM2.5_24h')
PM10_data<-pdata%>%filter(pdata$type=='PM10_24h')
SO2_data<-pdata%>%filter(pdata$type=='SO2_24h')
O3_data<-pdata%>%filter(pdata$type=='O3_24h')
AQI_data<-pdata%>%filter(pdata$type=='O3_24h')
AQI_data=AQI_data[,-2]
PM10_data=PM10_data[,-2]
PM2.5_data=PM2.5_data[,-2]
SO2_data=SO2_data[,-2]
O3_data=O3_data[,-2]
AQI_data
library(xlsx)
file.path("D:/R","PM2.5.xlsx")
write.table(PM2.5_data,file = "PM2.5.xlsx")
help("write.table")
PM2.5_data
#定义mutilpot函数
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  require(grid)
  
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots==1) {
    print(plots[[1]])
    
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}


#各个城市某一污染物趋势图
install.packages('MASS')
library(MASS)
help(box_cox)
help(rename)
dd<-AQI_data[,c(1,2)]%>%na.omit()%>%as_tsibble(index=date)
colnames(dd)<- c('x','y')
y=dd
y
b=boxcox(y~.,data=y)
l=which(b$y==max(b$y))
b$x[l]
b=boxcox(y~.,data=dd,lambda = 0.6)
y$y=box_cox(y$y,lambda=0.6)
plot_lst <- list()
y$year=year(y$x)
y$month=month(y$x)
y1=aggregate(y$y, by=list(y$year,y$month),mean)
y1_new<-y1[order(y1$Group.1,y1$Group.2),]
y1_new<-ts(y1_new,start=2015,end=2022,frequency = 12)
y1_new<-y1_new[,-1]
y1_new%>%gg_season()
autoplot(y1_new)
autoplot(y1_new)+theme(plot.title=element_text(hjust=0.5))+xlab("Time")+ylab("PM2.5")+labs(title = q)
install.packages('zoo')
library(zoo)
nm=colnames(dd)
q=nm[2]
y1_new

gg_subseries(y1_new)


plot_lst <- list()
for(i in c(2:dim(PM10_data)[2])){
  dd<-O3_data[,c(1,i)]%>%na.omit()%>%as_tsibble(index=date)  #处理数据不连续问题
  nm=colnames(dd)
  q=nm[2]
  colnames(dd)<- c('x','y')
  y=dd
  b=boxcox(y~.,data=y)     #BOX-COX
  l=which(b$y==max(b$y))
  lbd=b$x[l]               #最优lambda
  y$y=box_cox(y$y,lambda=lbd)
  y$year=year(y$x)
  y$month=month(y$x)
  y1=aggregate(y$y, by=list(y$year,y$month),mean)    #分组月度平均
  y1_new<-y1[order(y1$Group.1,y1$Group.2),]
  y1_new<-ts(y1_new,start=2015,end=2022,frequency = 12)
  y1_new<-y1_new[,-1]
  y1_new<-y1_new[,-1]
  y222=y1_new%>%as_tsibble()
  p1=gg_subseries(y222)+theme(plot.title=element_text(hjust=0.5))+xlab("Time")+ylab("O3")+labs(title = q)
  plot_lst[[i-1]] <- p1
}
Rmisc::multiplot(plotlist = plot_lst, cols = 4)                 #


plot_lst <- list()
for(i in c(2:dim(PM10_data)[2])){
  dd<-PM10_data[,c(1,i)]%>%na.omit()%>%as_tsibble(index=date)
  p1=autoplot(dd)+xlab("Time")+ylab("PM10")
  plot_lst[[i-1]] <- p1
}
Rmisc::multiplot(plotlist = plot_lst, cols = 4)

plot_lst <- list()
for(i in c(2:dim(SO2_data)[2])){
  dd<-SO2_data[,c(1,i)]%>%na.omit()%>%as_tsibble(index=date)
  p1=autoplot(dd)+xlab("Time")+ylab("SO2")
  plot_lst[[i-1]] <- p1
}
Rmisc::multiplot(plotlist = plot_lst, cols = 4)

plot_lst <- list()
for(i in c(2:dim(O3_data)[2])){
  dd<-O3_data[,c(1,i)]%>%na.omit()%>%as_tsibble(index=date)
  p1=autoplot(dd)+xlab("Time")+ylab("O3")
  plot_lst[[i-1]] <- p1
}
Rmisc::multiplot(plotlist = plot_lst, cols = 4)



y1_new=y1_new[,-1]
y1_new

gg_season(y1_new)
class(y1_new)
class(dd)

y222=data.frame(Y=as.matrix(y1_new), date=time(y1_new))
gg_season(y222)
dd<-PM2.5_data[,c(1,2)]%>%na.omit()%>%as_tsibble(index=date)
class(dd)
dd %>%tsibble::fill_gaps()%>%gg_subseries()
y222=y1_new%>%as_tsibble()
gg_season(y222)
#绘制季节图
dd<-AQI_data[,c(1,2)]%>%na.omit()%>%as_tsibble(index=date)
dd%>%tsibble::fill_gaps()%>%gg_season()
plot_lst <- list()
for(i in c(2:dim(PM2.5_data)[2])){
  dd<-AQI_data[,c(1,i)]%>%na.omit()%>%as_tsibble(index=date)
  p1=dd %>%tsibble::fill_gaps()%>%gg_season()
  plot_lst[[i-1]] <- p1
}
Rmisc::multiplot(plotlist = plot_lst, cols = 4)

plot_lst <- list()
for(i in c(2:dim(PM10_data)[2])){
  dd<-PM10_data[,c(1,i)]%>%na.omit()%>%as_tsibble(index=date)
  p1=dd %>%tsibble::fill_gaps()%>%gg_season()
  plot_lst[[i-1]] <- p1
}
Rmisc::multiplot(plotlist = plot_lst, cols = 4)

plot_lst <- list()
for(i in c(2:dim(SO2_data)[2])){
  dd<-SO2_data[,c(1,i)]%>%na.omit()%>%as_tsibble(index=date)
  p1=dd %>%tsibble::fill_gaps()%>%gg_season()
  plot_lst[[i-1]] <- p1
}
Rmisc::multiplot(plotlist = plot_lst, cols = 4)

plot_lst <- list()
for(i in c(2:dim(O3_data)[2])){
  dd<-O3_data[,c(1,i)]%>%na.omit()%>%as_tsibble(index=date)
  p1=dd %>%tsibble::fill_gaps()%>%gg_season()
  plot_lst[[i-1]] <- p1
}
Rmisc::multiplot(plotlist = plot_lst, cols = 4)

plot_lst <- list()
for(i in c(2:dim(O3_data)[2])){
  dd<-O3_data[,c(1,i)]%>%na.omit()%>%as_tsibble(index=date)
  p1=dd %>%tsibble::fill_gaps()%>%gg_season(polar=TRUE)
  plot_lst[[i-1]] <- p1
}
Rmisc::multiplot(plotlist = plot_lst, cols = 4)

#季节子序列图
dd
require(dplyr) 
require(lubridate) 
d1=dd %>% mutate(year = year(dd$date), monthnum = month(dd$date), month = month(dd$date))
d2=d1%>%group_by(year, month)
d3=d2%>%arrange(year, monthnum)%>%select(-monthnum)%>%select(-date)
#将数据按月分组求均值
d4=aggregate(d3$北京, by=list(type=d3$year,d3$month),mean)
d6=unite(d4,"date",type,Group.2 ,remove = TRUE)
d6$date=ymd6$date
d5=d6%>%as_tsibble(index=date)
d4
d7=as.Date(d6$date,'%Y_%m')
m1=d4%>%group_by(Group.2)
m1
d2
d3
