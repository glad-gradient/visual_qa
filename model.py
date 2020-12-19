import torch
import torchvision


class ImageEncoder(torch.nn.Module):
    def __init__(self, embedding_dim):
        super(ImageEncoder, self).__init__()
        pretrained_model = torchvision.models.resnet34(pretrained=True)
        in_features = pretrained_model.fc.in_features  # 512

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = torch.nn.Sequential(torch.nn.Linear(in_features, embedding_dim))

    def forward(self, image):
        features = self.model(image)
        l2_norm = torch.linalg.norm(features, ord=2, dim=1, keepdim=True).detach()
        features = features.div(l2_norm)
        return features


class QuestionEncoder(torch.nn.Module):
    def __init__(self, word_embeddings, word_embedding_dim, hidden_dim, num_layers, embedding_dim):
        super(QuestionEncoder, self).__init__()
        self.word_embeddings = torch.nn.Embedding.from_pretrained(word_embeddings, freeze=True)
        self.lstm = torch.nn.LSTM(
            input_size=word_embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = torch.nn.Linear(2 * num_layers * hidden_dim, embedding_dim)
        self.tanh = torch.nn.Tanh()

    def forward(self, question):
        embeds = self.word_embeddings(question)  # (batch_size, max_qst_length, word_embed_size=300)
        _, (hidden, cell) = self.lstm(embeds)  # (num_layers, batch_size, hidden_size=512)
        features = torch.cat((hidden, cell), dim=2)  # (num_layers, batch_size, 2*hidden_size)
        features = features.permute(1, 0, 2)  # (batch_size, num_layers, 2*hidden_size)
        features = features.reshape(features.size()[0], -1)  # (batch_size, 2*num_layers*hidden_size)
        features = self.fc(features)  # (batch_size, embedding_dim)
        features = self.tanh(features)

        return features


class VisualQAModel(torch.nn.Module):
    def __init__(self, word_embeddings, word_embedding_dim, answer_vocab_size, hidden_dim, num_layers, embedding_dim):
        super(VisualQAModel, self).__init__()
        self.image_encoder = ImageEncoder(embedding_dim)
        self.question_encoder = QuestionEncoder(word_embeddings, word_embedding_dim, hidden_dim, num_layers,
                                                embedding_dim)
        self.fc1 = torch.nn.Linear(embedding_dim, answer_vocab_size)
        self.fc2 = torch.nn.Linear(answer_vocab_size, answer_vocab_size)
        self.tanh = torch.nn.Tanh()
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, image, question):
        image_features = self.image_encoder(image)
        question_features = self.question_encoder(question)
        combined_embeds = torch.mul(image_features, question_features)
        combined_embeds = self.fc1(combined_embeds)
        combined_embeds = self.dropout(combined_embeds)
        combined_embeds = self.tanh(combined_embeds)
        combined_embeds = self.fc2(combined_embeds)
        combined_embeds = self.dropout(combined_embeds)
        combined_embeds = self.tanh(combined_embeds)

        return combined_embeds
