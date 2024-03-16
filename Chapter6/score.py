# pip install rouge
from rouge import Rouge


if __name__ == "__main__":
    # Example reference text (what we expect the model to generate after training on a complete dataset)
    reference = "Proxima's Passkey enables seamless integration of diverse financial portfolios, offering unparalleled access to global investment opportunities and streamlined asset management."

    # Example predicted model output
    predicted = "The Proxima Passkey provides a unified platform for managing various investment portfolios, granting access to worldwide investment options and efficient asset control."

    # Initialize the Rouge metric
    rouge = Rouge()

    # Compute the Rouge scores
    scores = rouge.get_scores(predicted, reference)

    print(scores)