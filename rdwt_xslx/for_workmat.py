import xlrd
from xlutils.copy import copy
import xlwt
xlsxfile = r'py8800_ballmap_20190809.xlsx'
sheet_name = "ballmap_20190809"
book = xlrd.open_workbook(xlsxfile)

wb = copy(book)
sheet = book.sheet_by_name(sheet_name)
new_sheet = wb.add_sheet('pinlist')

count = 0

for r in range(2, 44):
    print("+"*8)
    for c in range(1, 43):
        index = sheet.cell_value(r,0)+sheet.cell_value(1,c)
        value = sheet.cell_value(r, c)
        if index in ['A1', 'BB1', "BB42", "A42"]:
            continue
        new_sheet.write(count, 0, index)
        new_sheet.write(count, 1, value)
        count = count + 1
        
        print("{0:20}{1:20}".format(index, value))
#         print(sheet.cell_value(r,1)+ sheet.cell_value(0,c),sheet.cell_value(r, c))
#         print
        
wb.save("new_book.xlsx")
