Project_To_DO.pdf: Project_To_DO.tex cherubino.jpg
	latexmk -shell-escape Project_To_DO.tex -pdf
