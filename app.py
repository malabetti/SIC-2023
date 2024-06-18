from flask import Flask, render_template, request
from modelo import avaliar
from relatorio import criar_slide

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('./index.html')

@app.route('/modelo')
def modelo():
    return render_template('Brigadeirao.html')

@app.route('/predizer')
def predizer():
    return render_template('predicao.html')

@app.route('/resultado', methods=['POST'])
def resultado():
    if request.method == 'POST':
        # Obter os dados do formulário
        predicao_tipo = request.form['predicao_tipo']
        chocolate_50 = float(request.form['chocolate_50'])
        achocolatado = float(request.form['achocolatado'])
        manteiga = float(request.form['manteiga'])
        tempo_de_cozimento = float(request.form['tempo_de_cozimento'])

        x = avaliar(predicao_tipo, chocolate_50, achocolatado, manteiga, tempo_de_cozimento)

        #return f'Valor previsto para {predicao_tipo} foi "{x}"'
        return criar_slide(x, predicao_tipo)

if __name__ == '__main__':
    app.run()
