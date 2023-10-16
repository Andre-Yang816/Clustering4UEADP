library(ScottKnottESD)
library(readxl)
#Under the PictureData folder processed in the previous step, the folder with the obtained evaluation index values
#finalpath= "C:/Users/Andre/Desktop/CUDP-master/discussion/Result/new_dataset/"
finalpath= '../DrawPicData/'

#Folder to store sk results
#skresultpath='C:/Users/Andre/Desktop/CUDP-master/discussion/Result/picture_sk/'
skresultpath = '../picture_sk_new/'
#save picture
setwd(skresultpath)

file_names<- list.files(finalpath)
for (i in 1:length(file_names)) {
  #if (grepl(pattern = ".xlsx$",x = file_names[i]) )
  if (grepl(pattern = ".xlsx$",x = file_names[i]) )
  {
    print(file_names[i])
    name<-strsplit(file_names[i],split = ".xlsx")[[1]][1]
    #print(name)
    path=paste(finalpath,sep = "",file_names[i])
    print(path)
    csv<- read_excel(path)
    csv<-csv[-1]
    
    #sk <- sk_esd(csv)
    sk <- sk_esd(csv)
    #plot(sk)
    
    #par(mar=c(7,1,1,1)+1)
    #par(oma=c(7,3,3,3)) 
    future=paste('RQ1_Ranking_',name,sep = "",".jpg")
    #jpeg(file=future,width=1960,height=1080)
    jpeg(file=future, width=5000,height=2000, units = 'px',res=600)
    plot(sk,
         mar=c(7,1,1,1),
         las=2,
         #cex=0.6,
         #cex.axis=0.1,
         cex.lab=1.2,
         xlab = '',
         ylab = 'Rankings',
         #mgp = c(3,2,0),
         #xgap.axis = 3,
         #axis.line=3,
         family = "serif",
         title=NULL
        )
    dev.off()
    
    resultpath=paste(skresultpath,sep = "",file_names[i])
    resultpath=paste(resultpath,sep = "",".txt")
    print(resultpath)
    
    #write.table (sk[["groups"]], resultpath)
    graphpath = paste(skresultpath,sep = "",file_names[i])
    #graphpath = paste(graphpath,sep = "",".jpg")
  }
}