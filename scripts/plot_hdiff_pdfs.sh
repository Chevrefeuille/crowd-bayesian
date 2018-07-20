#!/usr/bin/gnuplot
#!/opt/local/bin/gnuplot


# wxt
#set terminal aqua
set terminal wxt size 480,360 enhanced font 'Verdana,10' persist


set autoscale                     

set grid ytics lt 0 lw 2 lc rgb "#bbbbbb"
set grid xtics lt 0 lw 2 lc rgb "#bbbbbb"

set border linewidth 1.5

# color definitions
set  linestyle 1 linecolor rgb '#0060ad' lt 1 lw 5 pt 7 ps 0.3   # --- blue
set  linestyle 2 linecolor rgb '#ad0009' lt 1 lw 5 pt 7 ps 0.3   # --- red
set  linestyle 3 linecolor rgb '#008000' lt 1 lw 5 pt 7 ps 0.3   # --- green
set  linestyle 4 linecolor rgb '#ffa500' lt 1 lw 5 pt 7 ps 0.3   # --- orange


set ytics 0.002
set tics scale 0.75

set xrange [0:600]
set yrange [0:0.01]
#set xtics ( "-3" -3000, "-1.5" -1500, "0" 0, "1.5" 1500, "3" 3000)

set xlabel "{/Symbol D}_{/Symbol h}(mm)"
set ylabel "p({/Symbol D}_{/Symbol h})"

plot '../data/pdfs/h_diff_doryo.txt' with lines  ls 1  title "C", \
     '../data/pdfs/h_diff_koibito.txt' with lines  ls 2  title "M", \
    #  '../data/pdfs/h_diff_kazoku.txt' with lines  ls 3  title "A", \
    # '../data/pdfs/h_diff_yujin.txt' with lines  ls 4  title "R"
set key right top

set terminal pdf
set output '../results/hdiff_pdfs.pdf'
# set output '../results/hdiff_pdfs_all.pdf'
replot 
set terminal x11
