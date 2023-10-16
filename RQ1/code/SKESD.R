library(ScottKnottESD)
library(readxl)
finalpath= "../DrawPicData/"
skresultpath='../output/'
file_names<- list.files(finalpath)
for (i in 1:length(file_names)) {
    
    path=paste(finalpath,sep = "",file_names[i])
    print(path)
    #csv<- read.csv(file=path, header=TRUE, sep=",")
    csv<- read_excel(path)
    csv<-csv[-1]
    sk <- sk_esd(csv)
    
    
    #plot(sk)
    
    resultpath=paste(skresultpath,sep = "",file_names[i])
    resultpath=paste(resultpath,sep = "",".txt")
    print(resultpath)
    
    write.table (sk[["groups"]], resultpath) 
}
