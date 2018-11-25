data<-read.csv ("Dataset/credit_default_cleaned.csv", header=TRUE)
sort.order<-order(data$from)
data<-data[sort.order,]
num.rows = nrow(data)
one.before.last.row = num.rows-1
decimal.places=4
bar.names<-paste("[", round(data$from[1:one.before.last.row], decimal.places), ", ", round (data$to[1:one.before.last.row], decimal.places), ")")
bar.names<-c(bar.names, paste("[", round (data$from[num.rows], decimal.places) , ", ", round (data$to[num.rows], decimal.places), "]"))
par (mar=c(6, 11, 2, 2))
barplot (data$value, names.arg=bar.names, las=2, horiz=TRUE)
