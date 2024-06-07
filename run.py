import tkinter as tk
from tkinter import filedialog, ttk
from model import Task
from tkinter import messagebox


class Run:
    def __init__(self):
        root = tk.Tk()
        root.title("图片识别")

        style = ttk.Style(root)
        style.theme_use('clam')  # 更现代的主题

        # 设置背景颜色
        background_color = '#FFFFFF'
        root.configure(bg=background_color)

        # 应用于所有标准ttk控件的样式
        style.configure('TButton', background=background_color)
        style.configure('TRadiobutton', background=background_color)
        style.configure('TCheckbutton', background=background_color)
        style.configure('TLabel', background=background_color)
        style.map('TButton', background=[('active', background_color)])

        self.folder_selected = tk.StringVar()
        self.file_selected = tk.StringVar()
        self.model_type = tk.StringVar()

        ttk.Label(root, text="选择图片文件夹:", background=background_color).grid(row=0, column=0, padx=20, pady=10,
                                                                                  sticky='w')
        folder_button = ttk.Button(root, text="浏览...", command=self.choose_folder)
        folder_button.grid(row=0, column=1, padx=10, pady=10, sticky='ew')
        ttk.Label(root, textvariable=self.folder_selected, background=background_color).grid(row=0, column=2, padx=10,
                                                                                             pady=10, sticky='ew')

        ttk.Label(root, text="选择目标文件:", background=background_color).grid(row=1, column=0, padx=20, pady=10,
                                                                                sticky='w')
        file_button = ttk.Button(root, text="浏览...", command=self.choose_file)
        file_button.grid(row=1, column=1, padx=10, pady=10, sticky='ew')
        ttk.Label(root, textvariable=self.file_selected, background=background_color).grid(row=1, column=2, padx=10,
                                                                                           pady=10,
                                                                                           sticky='ew')

        ttk.Label(root, text="选择模型类型:", background=background_color).grid(row=2, column=0, padx=20, pady=10,
                                                                                sticky='w')
        ttk.Radiobutton(root, text="KNN", variable=self.model_type, value="knn").grid(row=2, column=1, sticky='w',
                                                                                      padx=(10, 10))
        ttk.Radiobutton(root, text="CNN", variable=self.model_type, value="cnn").grid(row=3, column=1, sticky='w',
                                                                                      padx=(10, 10))

        run_button = ttk.Button(root, text="运行模型", command=self.run_model)
        run_button.grid(row=4, column=0, columnspan=5, padx=20, pady=10, sticky='ew')

        root.mainloop()

    def choose_folder(self):
        self.folder_selected.set(filedialog.askdirectory())

    def choose_file(self):
        self.file_selected.set(filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx;*.xls")]))

    def run_model(self):
        # 收集用户输入数据
        image_folder = self.folder_selected.get()
        target_file = self.file_selected.get()
        model_choice = self.model_type.get()

        # 创建Run实例
        run_instance = Task(image_path=image_folder, target_path=target_file, mode=model_choice)
        run_instance.run()

        messagebox.showinfo("完成", "运行结束")


if __name__ == '__main__':
    instance = Run()
