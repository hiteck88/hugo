library(quantmod)
options(warn = -1)
library(reticulate)
#use_condaenv( "py37")
#use_virtualenv("/home/qi/environments/my_env/bin/python3")
#font.add('kaishu','simhei.ttf')
library(reticulate)
library(tidyverse)
library(quantmod)
library(timetk)
library(formattable)
library(kableExtra)
QA <- import('QUANTAXIS')
Qi <- import('qi_library_qa') 
today = Sys.Date()


Sys.setlocale(, 'Chinese')
#Sys.setlocale("LC_ALL","zh_CN.utf-8")
#options(shiny.usecairo = FALSE)
#out = Qi$QA_filter_day(0.09,0,0,0)
#codelist = out[[1]]
#namelist = out[[2]]
codelist =  c('603883')

myTheme<-chart_theme()
myTheme$col$up.col<-'darkgreen'
myTheme$col$dn.col<-'darkred'
myTheme$col$dn.border <- 'black'
myTheme$col$up.border <- 'black'
myTheme$rylab <- FALSE
myTheme$col$grid <- "lightgrey"

customGreen0 = "#DeF7E9"

customGreen = "#71CA97"

customRed = "#ff7f7f"

improvement_formatter <- formatter("span", 
                                   style = x ~ formattable::style(font.weight = "bold", 
                                                     color = ifelse(x > 0, customGreen, ifelse(x < 0, customRed, "black"))), 
                                   x ~ icontext(ifelse(x>0, "arrow-up", "arrow-down"), x)
)

output.stock <- function(code){
  df1 = Qi$QA_fetch_data_day_adv(code, '2019-6-1',today)
  #data = df1$data
  #data$date = df1$datetime$values
  df = tk_xts(df1)
  #colnames(df) <- c('Open', 'High', 'Low', 'Close', 'Volume', 'Amount')
  chart_Series(df,subset='2019-6::2019-11',TA="add_Vo();add_BBands();add_SMA(5);add_MACD()", theme = myTheme, name= paste("日", code))
  
}

output.stockWeek <- function(code){
  df1 = Qi$QA_resample_day_R(code, '2019-1-1',today, 'W')
  df = tk_xts(df1)
  #colnames(df) <- c('Open', 'High', 'Low', 'Close', 'Volume', 'Amount')
  chart_Series(df,subset='2019-1::2019-11',TA="add_Vo();add_SMA(5)", theme = myTheme,name= paste("周",code))
}

output.stockMonth <- function(code){
  df1 = Qi$QA_resample_day_R(code, '2018-1-1',today, 'M')
  #data = Qi$QA_resample_day_R(df1, 'M')
  df = tk_xts(df1)
  #colnames(df) <- c('Open', 'High', 'Low', 'Close', 'Volume', 'Amount')
  chart_Series(df,subset='2018-1::2019-11',TA="add_Vo();add_SMA(5)",theme = myTheme,name= paste("月",code))
  }

output.gapTable <- function(code){
    gap_data = Qi$gap_list(code)[c('date','close2','up_count','down_count','diff','current_close')] 
    #gap_data$date = as.Date(gap_data$date)
    gap_data$date <- format(gap_data$date,'%Y-%m-%d')
    names(gap_data) <- c("date", "close","up","down","diff","price")
    gap_data$up = as.integer(gap_data$up)
    gap_data$down = as.integer(gap_data$down)
    
    out2 <- Qi$get_ma_ouptput(c(tuple(code, code)))
    out2 <- out2[c("date","ma5", "type","diff","close")]
    names(out2) <- c("date", "close", "up","diff","price")
    out2$down = 0
   
    
    out3 <- rbind(gap_data,out2) %>% arrange(diff) 
    out3$diff = round(out3$diff, 2)
    out3$close = round(out3$close, 2)
    names(out3) <- c("日期", "缺口价格", "向上缺口","向下缺口","差价","现价")
    
    formattable(out3, align =c("l","c","c","c","c", "c"), list(
      `日期` = formatter("span", style = ~ formattable::style(color = "grey",font.weight = "bold")), 
      `缺口价格`= color_tile(customGreen, customGreen0),
      `向上缺口`= color_tile(customGreen, customGreen0),
      `向下缺口`= color_tile(customGreen, customGreen0),
      `现价` = color_bar(customRed),
      `差价` = improvement_formatter
    ))
    
    
}