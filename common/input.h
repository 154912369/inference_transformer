#include "sentencepiece_processor.h"
class Input{
    private:
    sentencepiece::SentencePieceProcessor processor;
    bool first= true;
    public:
    Input(std::string model_path);
    bool get_input(std::vector<std::string>& pieces);
};