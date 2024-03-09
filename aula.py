def decifrar_cesar(cifra, deslocamento):
    resultado = ""

    for char in cifra:
        # Verifica se o caractere é uma letra
        if char.isalpha():
            # Obtém o código ASCII do caractere
            codigo_ascii = ord(char)

            # Determina a nova posição após o deslocamento
            nova_posicao = codigo_ascii - deslocamento

            # Verifica se o caractere era maiúsculo ou minúsculo
            if char.isupper():
                # Garante que a nova posição esteja dentro do alfabeto
                nova_posicao = (nova_posicao - ord('A')) % 26 + ord('A')
            else:
                nova_posicao = (nova_posicao - ord('a')) % 26 + ord('a')

            # Converte a nova posição de volta para caractere
            caractere_decifrado = chr(nova_posicao)

            # Adiciona o caractere decifrado ao resultado
            resultado += caractere_decifrado
        else:
            # Se não for uma letra, adiciona o caractere original
            resultado += char

    return resultado

# Exemplo de uso
cifra_cesar = "z#zigv#wz#tfviiz#klwv#hvi#fhzwz#kziz#mlhhlh#mvtlxrlh"
deslocamento = 11
mensagem_decifrada = decifrar_cesar(cifra_cesar, deslocamento)

print("Cifra de César:", cifra_cesar)
print("Mensagem decifrada:", mensagem_decifrada)