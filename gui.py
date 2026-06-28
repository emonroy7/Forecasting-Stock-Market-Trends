# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
import matplotlib.pyplot as plt  # Import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import io
import sys
import stock_analysis
import trading_strategy
import risk_analysis
import portfolio_analysis

class StockAnalysisApp:
    def __init__(self, master):
        self.master = master
        master.title("Stock Analysis App")

        self.tabControl = ttk.Notebook(master)
        self.tabControl.pack(expand=1, fill="both")

        self.tab1 = ttk.Frame(self.tabControl)
        self.tabControl.add(self.tab1, text="Stock Analysis")
        self.tab2 = ttk.Frame(self.tabControl)
        self.tabControl.add(self.tab2, text="Trading Strategy")
        self.tab3 = ttk.Frame(self.tabControl)
        self.tabControl.add(self.tab3, text="Risk Analysis")
        self.tab4 = ttk.Frame(self.tabControl)
        self.tabControl.add(self.tab4, text="Portfolio Analysis")

        self.create_tab1_widgets()
        self.create_tab2_widgets()
        self.create_tab3_widgets()
        self.create_tab4_widgets()

    def create_tab1_widgets(self):
        self.tab1_label = ttk.Label(self.tab1, text="Stock Analysis", font=("Helvetica", 18))
        self.tab1_label.grid(row=0, column=0, padx=10, pady=10)

        self.tab1_output_text = scrolledtext.ScrolledText(self.tab1, width=60, height=10, wrap=tk.WORD)
        self.tab1_output_text.grid(row=1, column=0, padx=10, pady=10)

        self.tab1_button = ttk.Button(self.tab1, text="Run Analysis", command=self.run_stock_analysis)
        self.tab1_button.grid(row=2, column=0, padx=10, pady=10)

    def run_stock_analysis(self):
        # Redirect stdout to capture output
        output_capture = io.StringIO()
        sys.stdout = output_capture

        stock_analysis.main()

        # Restore stdout
        sys.stdout = sys.__stdout__

        # Display output in GUI
        self.tab1_output_text.insert(tk.END, output_capture.getvalue())

    def create_tab2_widgets(self):
        self.tab2_label = ttk.Label(self.tab2, text="Trading Strategy", font=("Helvetica", 18))
        self.tab2_label.grid(row=0, column=0, padx=10, pady=10)

        self.tab2_output_text = scrolledtext.ScrolledText(self.tab2, width=60, height=10, wrap=tk.WORD)
        self.tab2_output_text.grid(row=1, column=0, padx=10, pady=10)

        self.tab2_button = ttk.Button(self.tab2, text="Run Strategy", command=self.run_trading_strategy)
        self.tab2_button.grid(row=2, column=0, padx=10, pady=10)

        self.fig = plt.Figure(figsize=(8, 6), dpi=100)  # Create a Figure object
        self.plot = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tab2)
        self.canvas.get_tk_widget().grid(row=3, column=0, padx=10, pady=10)

    def run_trading_strategy(self):
        # Redirect stdout to capture output
        output_capture = io.StringIO()
        sys.stdout = output_capture

        trading_strategy.main()

        # Restore stdout
        sys.stdout = sys.__stdout__

        # Display output in GUI
        self.tab2_output_text.insert(tk.END, output_capture.getvalue())

        # Plot graph
        data = trading_strategy.fetch_stock_data("AAPL", "2010-01-01", "2022-01-01")
        data = trading_strategy.calculate_technical_indicators(data)

        self.plot.clear()
        self.plot.plot(data.index, data['Close'], label='Close Price', color='black')
        self.plot.plot(data.index, data['MA50'], label='50-day MA', color='blue')
        self.plot.plot(data.index, data['MA200'], label='200-day MA', color='red')
        self.plot.set_title('Historical Price and Moving Averages')
        self.plot.set_xlabel('Date')
        self.plot.set_ylabel('Price')
        self.plot.legend()

        self.canvas.draw()

    def create_tab3_widgets(self):
        self.tab3_label = ttk.Label(self.tab3, text="Risk Analysis", font=("Helvetica", 18))
        self.tab3_label.grid(row=0, column=0, padx=10, pady=10)

        self.tab3_output_text = scrolledtext.ScrolledText(self.tab3, width=60, height=10, wrap=tk.WORD)
        self.tab3_output_text.grid(row=1, column=0, padx=10, pady=10)

        self.tab3_button = ttk.Button(self.tab3, text="Run Analysis", command=self.run_risk_analysis)
        self.tab3_button.grid(row=2, column=0, padx=10, pady=10)

    def run_risk_analysis(self):
        # Redirect stdout to capture output
        output_capture = io.StringIO()
        sys.stdout = output_capture

        risk_analysis.main()

        # Restore stdout
        sys.stdout = sys.__stdout__

        # Display output in GUI
        self.tab3_output_text.insert(tk.END, output_capture.getvalue())

    def create_tab4_widgets(self):
        self.tab4_label = ttk.Label(self.tab4, text="Portfolio Analysis", font=("Helvetica", 18))
        self.tab4_label.grid(row=0, column=0, padx=10, pady=10)

        self.tab4_output_text = scrolledtext.ScrolledText(self.tab4, width=60, height=10, wrap=tk.WORD)
        self.tab4_output_text.grid(row=1, column=0, padx=10, pady=10)

        self.tab4_button = ttk.Button(self.tab4, text="Run Analysis", command=self.run_portfolio_analysis)
        self.tab4_button.grid(row=2, column=0, padx=10, pady=10)

    def run_portfolio_analysis(self):
        # Redirect stdout to capture output
        output_capture = io.StringIO()
        sys.stdout = output_capture

        portfolio_analysis.main()

        # Restore stdout
        sys.stdout = sys.__stdout__

        # Display output in GUI
        self.tab4_output_text.insert(tk.END, output_capture.getvalue())

def main():
    root = tk.Tk()
    app = StockAnalysisApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()