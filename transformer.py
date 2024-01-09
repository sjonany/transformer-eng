"""
Python implementation of decoder-only transformer that closely follows
https://blog.nelhage.com/post/transformers-for-software-engineers/

Like the original post, this is missing: [positional embedding, layer normalization, pre-softmax scaling in attention]
"""
# Allows self-reference of class in typing hint.
# https://stackoverflow.com/a/36193829/21196296
from __future__ import annotations
from typing import Annotated, Callable, Dict
import numpy as np
import numpy.typing as npt

"""Type annotations"""
NDArrayInt = npt.NDArray[np.int_]
NDArrayFloat = npt.NDArray[np.float64]

"""Constants"""
# The number of residual blocks. It's called N_LAYERS in the original post
# Each residual block is a sequence of attention, MLP, and normalization layers
# In this implementation, we just have the attention and MLP layer, and skip layer normalization.
# Note: In the blog post this 96
N_BLOCKS = 4

# Token embedding dimension. It's called D_MODEL in the original post.
# This is also `State`'s dimension. -- State is an item in the residual stream.
# Note: In the blog post this is 12288
D_EMBED = 512

# The number of neurons in the hidden layer of MLP.
D_MLP = 4 * D_EMBED

# The smaller dimension of the attention-score-weighted value vectors.
# It's D_HEAD in the original post.
D_ATTENTION = 128

# The dimension of query and key vectors. It doesn't have to be D_ATTENTION, but in practice it is.
D_ATTENTION_QUERY = D_ATTENTION

# Number of attention heads
# All the N_HEADS summary vectors can be concatenated and re-projected back to D_EMBED,
# so it actually doesn't have to be D_EMBED / D_ATTENTION, but in practice it is.
# Note: In the blog post this is 96
N_HEADS = 4
assert N_HEADS == D_EMBED / D_ATTENTION

# The number of input tokens the transformer can handle at once.
# Also the number of `State`s in the residual stream.
# Note: In practice this is ~1024
N_TOKEN = 16

# The number of unique subwords
# Note: In the blog post this is 50000
N_VOCAB = 100

class Transformer:

    token_to_state_embedder: TokenToStateEmbedder
    # residual blocks
    blocks: Annotated[list[Block], N_BLOCKS]
    state_to_token_logits_unembedder: StateToTokenLogitsUnembedder

    def __init__(self, token_to_state_embedder, state_to_token_logits_unembedder, blocks):
        self.token_to_state_embedder = token_to_state_embedder
        self.state_to_token_logits_unembedder = state_to_token_logits_unembedder
        self.blocks = blocks

    def run(self, input_tokens: Annotated[list[TokenId], N_TOKEN]) -> Annotated[list[Logits], N_TOKEN]:
        """A transformer accepts an ordered list of subwords / tokens and
        returns, for each prefix of tokens, the logits of the next subword.
        Type: TokenId[N_TOKEN] -> Logits[N_TOKEN]

        [For each prefix] It's important to note that the return value is NOT just the logits for the next subword following the
        ENTIRE context / sentence, but rather, the logits of the next subword FOR EACH non-empty prefix of the input.
        E.g. if the input is "this is my sentence", then we will return 4 logits. The first logit is for the next subword after "this",
        whereas the fourth / last logit is for the next subword after "this is my sentence".
        This might look silly at first, since you might think you only need the final logit,
        but it's useful to get multiple loss numbers for training.

        [Positioning] The transformer doesn't actually know about the position of the subwords. We get around this by
        adding positional embedding into each token embedding. **But we don't do it in this exercise.
        """

        # For each token, we embed into a D_EMBED vector.
        states = [self.token_to_state_embedder.run(token_id) for token_id in input_tokens]

        # The initial residual stream is just the token embeddings.
        residual_stream = ResidualStream(states)

        # A lot of heavy processing on the residual stream by going through a sequence of blocks.
        # The meaning of the residual stream changes after each block.
        for block in (self.blocks):
            # There are two stages:
            # 1. Mix states / opaque data structures using attention
            # 2. Process each mixture of states independently using MLP

            # Runtime: O(N_TOKEN * N_HEAD * D_ATTENTION * (N_TOKEN + D_EMBED))
            # = O(N_TOKEN * D_EMBED * (N_TOKEN + D_EMBED))
            attention_layer_update = block.attention_layer.run(residual_stream)
            residual_stream = residual_stream.apply_update(attention_layer_update)

            # At each layer, the same MLP object is applied to each state in the residual stream individually.
            # Runtime: O(N_TOKEN * D_EMBED * D_MLP)
            new_states = residual_stream.states
            for state_i in range(len(residual_stream.states)):
                state_update = block.mlp_layer.run(new_states[state_i])
                new_states[state_i] = new_states[state_i].apply_update(state_update)

            # Overall runtime = O(N_TOKEN * D_EMBED * (N_TOKEN + D_MLP))
            residual_stream = ResidualStream(new_states)

        # At this point, the residual stream is N_TOKENS states.
        # Each state is a summary of a prefix for the input.
        # That is, states[2] is a summary of "I love you", whereas states[1] is a summary of "I love".
        # This happens because in the attention layers, we ensure that states[i] will never depend on
        # the initial token embeddings of tokens after i.
        logits_for_each_prefix = [self.state_to_token_logits_unembedder.run(prefix_summary_state)\
                                  for prefix_summary_state in residual_stream.states]
        return logits_for_each_prefix


class ResidualStream:
    """A snapshot of transformer's internal state at a specific layer. Type: State[N_TOKEN] 
    You can think of this as a N_TOKEN-size dictionary of opaque data structures.
    
    Note the meanings of this State dictionary changes after each layer.
    E.g. After the initial embedding layer, ResidualStream is a collection of token embeddings.
    After the first attention step, this is a set of context-aware embeddings per token.
    After the first MLP, it's something mysterious :)
    
    So, beware of the name ResidualStream. Conceptually they're so different they might as well be called
    ResidualStreamLayer1, ResidualStreamLayer2, ...
    """
    states: Annotated[list[State], N_TOKEN]
    def __init__(self, states):
        self.states = states
    

    def apply_update(self, update: ResidualStreamUpdate) -> ResidualStream:
        # Apply update to each state independently.
        # Each state update is just a vector addition.
        new_states = []
        for state_i in range(N_TOKEN):
            new_states.append(self.states[state_i].apply_update(update.state_updates[state_i]))
        return ResidualStream(new_states)


class State:
    """An item in the residual stream. Type: float[D_EMBED]
    In the initial residual stream, each State is exactly a single token embedding.
    For the later residual streams, the State can be thought as an opaque data structure.

    We should NOT expect dimension state.data[i] to have a special meaning.
    Instead, meaning is probably encoded in the (almost) _basis_ vectors of the state.
    Remember that to get a "field" in this "opaque data structure" what we do is take dot products
    to get a scalar.
    I also liked the blog post's note on how this is an efficient representation that can encode
    many more than just D_EMBED scalars (exponentially more in fact).
    """
    data: Annotated[NDArrayFloat, D_EMBED]

    def __init__(self, data):
        self.data = data
        
    def apply_update(self, update: StateUpdate) -> State:
        """Apply an update to this state and return as a new state.
        """
        return State(self.data + update.data)


class ResidualStreamUpdate:
    """An update to the residual stream. Type: StateUpdate[N_TOKEN]
    E.g.
    Update =
        [StateUpdate=[0.1,0.2], StateUpdate=[0.3,0.4]]
    applied to the residual stream of
        [State=[1,1], State=[1,1]]
    means residual stream will become
        [State=[1.1,1.2], State=[1.3,1.4]]
    """
    state_updates: Annotated[list[StateUpdate], N_TOKEN]

    def __init__(self, state_updates):
        self.state_updates = state_updates

    def add(self, update: ResidualStreamUpdate) -> ResidualStreamUpdate:
        """Add another residual stream update to this one.
        """
        if update is None:
            return self
        return ResidualStreamUpdate([self.state_updates[i].add(update.state_updates[i]) for i in range(N_TOKEN)])


class StateUpdate:
    """An update to a single state. Type: float[D_EMBED]
    E.g. Update = [0.1, 0.2] to the 5th state [1.0, 2.0] in the residual stream
    means the residual stream will have 5th state become [1.1, 1.2]
    """
    data: Annotated[NDArrayFloat, D_EMBED]

    def __init__(self, data):
        self.data = data

    def add(self, update: StateUpdate) -> StateUpdate:
        """Add another state update to this one.
        """
        return StateUpdate(self.data + update.data)


class TokenToStateEmbedder:
    """Convert a token ID to a D_EMBED vector."""
    # Ther are N_VOCAB entries in the embedding matrix
    embedding_matrix: Dict[TokenId, Annotated[NDArrayFloat, D_EMBED]]

    def __init__(self, embedding_matrix):
        self.embedding_matrix = embedding_matrix

    def run(self, input_token: TokenId) -> State:
        return State(self.embedding_matrix[input_token])


class TokenId:
    """Which subword. Type: int
    E.g. 'cake' subword can correspond to token ID 123."""
    data: int
    def __init__(self, data):
        assert data < N_VOCAB and data >= 0
        self.data = data
    
    def __hash__(self):
        return hash(self.data)
    
    def __eq__(self, rhs):
        return isinstance(rhs, TokenId) and self.data == rhs.data


class Logits:
    """The logit for each subword. Type: float[N_VOCAB]
    Logits get softmax'ed to get a probability distribution of the next subword.
    E.g. If the softmax'ed logit vector gives [0.1, 0.2, 0.7], that then means the next subword is
    the token corresponding to token ID 2 (E.g. 'cake') with 70% probability.
    """
    data: Annotated[NDArrayFloat, N_VOCAB]
    def __init__(self, data):
        self.data = data


class StateToTokenLogitsUnembedder:
    """Convert a state to logits.
    Various views for unembedding:
    (1 - what's being coded) A list of functions, one per vocab item.
       Each takes a state, and returns 1 logit for that vocab.
       E.g. a single function for "cake" measures how "cake"-like this state is, and returns a scalar.
    (2) A list of D_EMBED-dimensional vectors, one for each vocab item.
       You then dot-product these D_EMBED vectors with the state to get how aligned the state vector
       is with the vocab item.
    (3) A matrix of shape (N_VOCAB, D_EMBED). Each row can be dot-product-ed with the state to get
    a measure of how vocab-item-like the state is.
        Remember: (N_VOCAB, D_EMBED) X (D_EMBED, 1) = (N_VOCAB, 1)
    """
    logit_computer_per_vocab: Dict[TokenId, LogitFn]


    def __init__(self, logit_computer_per_vocab):
        self.logit_computer_per_vocab = logit_computer_per_vocab

    def run(self, state: State) -> Logits:
        """Convert 1 state of the final residual stream to logits of the next subword.
        Each state is kind of like a summary of 1 prefix of the input context / sentence.

        E.g. given a state summarizing the substring "I love you", return these logits for the following subword:
          "more": 10
          "dear": 10
          "you": 0
        """
        logit_per_vocab = []
        for vocab_i in range(N_VOCAB):
            logit_per_vocab.append(self.logit_computer_per_vocab[TokenId(vocab_i)].run(state))
        return logit_per_vocab


class LogitFn:
    """Takes a state and returns a logit for a specific vocab item.
    E.g. LogitFn `cake` is a measure of how cake-like the state is."""

    # This vector, when dot-producted with a state,
    # returns a measure of how attuned the state is to this vocab.
    vocab_likeness_vector: Annotated[NDArrayFloat, D_EMBED]

    def __init__(self, vocab_likeness_vector):
        self.vocab_likeness_vector = vocab_likeness_vector
    
    def run(self, state: State) -> float:
        return np.dot(self.vocab_likeness_vector, state.data)


class Block:
    """A residual block processes a residual stream and returns one with a new meaning.
    It's composed of an attention layer and an MLP layer. We omit layer normalization."""

    attention_layer: AttentionLayer
    mlp_layer: MLPLayer

    def __init__(self, attention_layer, mlp_layer):
        self.attention_layer = attention_layer
        self.mlp_layer = mlp_layer

class AttentionLayer:
    """The attention layer takes a residual stream, and returns an update that takes into account
    cross-state interactions."""

    attention_heads: Annotated[list[AttentionHead], N_HEADS]

    def __init__(self, attention_heads):
        self.attention_heads = attention_heads

    def run(self, residual_stream: ResidualStream) -> ResidualStreamUpdate:
        # Each head is applied independently to the residual stream
        # Then we simply sum them up 
        all_head_residual_stream_update = None
        for head in self.attention_heads:
            per_head_residual_stream_update = head.run(residual_stream)
            all_head_residual_stream_update = per_head_residual_stream_update.add(all_head_residual_stream_update)
        return all_head_residual_stream_update

class AttentionHead:
    """A single attention head also takes a residual stream and returns an update.
    Each head has its own Q,K,V lgoic"""
    # A single attention head has a single Q,K,V logic.

    # How you convert a state to to q,k,v vectors.
    q_projector = Annotated[NDArrayFloat, D_ATTENTION_QUERY, D_EMBED]
    k_projector = Annotated[NDArrayFloat, D_ATTENTION_QUERY, D_EMBED]
    v_projector = Annotated[NDArrayFloat, D_ATTENTION, D_EMBED]
    attention_to_state_projector = Annotated[NDArrayFloat, D_EMBED, D_ATTENTION]

    def __init__(self, q_projector, k_projector, v_projector, attention_to_state_projector):
        self.q_projector = q_projector
        self.k_projector = k_projector
        self.v_projector = v_projector
        self.attention_to_state_projector = attention_to_state_projector

    def run(self, residual_stream: ResidualStream) -> ResidualStreamUpdate:
        ## Precompute lower-D space representations for each state.

        # For each state, precompute 3 vectors: q(state), k(state), v(state)
        # qs_per_state has N_TOKEN vectors, each of size D_ATTENTION_QUERY
        qs_per_state = [self.q_projector @ state.data for state in residual_stream.states]
        ks_per_state = [self.k_projector @ state.data for state in residual_stream.states]
        # vs_per_state has N_TOKEN vectors, each of size D_ATTENTION
        vs_per_state = [self.v_projector @ state.data for state in residual_stream.states]

        state_updates = []
        ## Compute each state update independently. This is where the state-to-state mixing happens.
        # Note this loop is what people mean when they say "quadratic performance"
        # For each head, it's O(N_TOKEN * D_ATTENTION * (N_TOKEN + D_EMBED))

        for target_state_i in range(N_TOKEN):
            # I can only look at states before and including me to concoct the state update.
            states_to_mix = residual_stream.states[0:target_state_i+1]
            # how_much_target_cares[j] is how much states[target_state_i] cares about states[j]
            how_much_target_cares = []
            for k_state_j in range(len(states_to_mix)):
                how_much_target_cares.append(np.dot(qs_per_state[target_state_i], ks_per_state[k_state_j]))
            how_much_target_cares = softmax(how_much_target_cares)

            # Given how much we care, compute the context / summary vector that the target state
            # should use to compute its update
            weighted_vs = np.zeros(D_ATTENTION)
            for v_state_i in range(len(states_to_mix)):
                weighted_vs += how_much_target_cares[v_state_i] * vs_per_state[v_state_i]

            # Reproject from D_ATTENTION to D_EMBED
            state_update = StateUpdate(self.attention_to_state_projector @ weighted_vs)
            state_updates.append(state_update)
        return ResidualStreamUpdate(state_updates)
        
class MLPLayer:
        """In each layer, MLP is applied to each state independently and returns a state update."""
        mlp_neurons: Annotated[list[MLPNeuron], D_MLP]

        def __init__(self, mlp_neurons) -> None:
            self.mlp_neurons = mlp_neurons

        def run(self, state: State) -> StateUpdate:
            """MLP layer is applied to each state in the residual stream individually.
            Given a state, it returns a state update.
            
            Here are various ways of looking at the MLP layer:
            (1) Matrix. We start with residual stream of shape (N_TOKEN, D_EMBED).
                (a) We multiply (N_TOKEN, D_EMBED) by (D_EMBED, D_MLP) to get (N_TOKEN, D_MLP)
                (b) Apply non-linearity element wise
                (c) We multiply (N_TOKEN, D_MLP) by (D_MLP, D_EMBED) so we can add back to residual stream
            (2) Per state. We map the (1) Matrix view to a per-state view by referring to the alphabet steps (a), (b), ...
                - We have N_TOKEN state vectors, each of size D_EMBED.
                (a) We multiply (N_TOKEN, D_EMBED) by (D_EMBED, D_MLP) to get (N_TOKEN, D_MLP)
                  - Each D_EMBED state vector is being processed independently by the same set of D_MLP neurons.
                    We now discuss what happens to each D_EMBED state vector.
                  - We convert the D_EMBED state vector to a D_MLP vector by feeding it into a set of D_MLP neurons.
                    - Each neuron takes a D_EMBED vector and returns a scalar independently
                      - It is able to do so because each neuron has its own D_EMBED vector to do dot product with.
                      This inner vector, in a sense, is what the neuron is looking for from a single state.
                      - To relate to interpretation (1), each neuron is a column vector in the (D_EMBED, D_MLP) matrix.
                    - Now that we have D_MLP scalars, we just concatenate them.
                (b) Apply non-linearity element wise
                  - Each of the D_MLP scalars is then passed through non-linearity in an element-wise fashion.
                (c) We multiply (N_TOKEN, D_MLP) by (D_MLP, D_EMBED) so we can add back to residual stream
                  - Each of the D_MLP neurons knows how much to use its 1 scalar to contribute to each of the D_EMBED scalars
                  - It does so because internally it has another D_EMBED vector to figure out how to distribute its scalar
                    to the final state update vector.
            """
            # Run each neuron independently on the state, then sum up the state updates.
            final_state_update = np.zeros(D_EMBED)
            for neuron in self.mlp_neurons:
                state_update = neuron.run(state)
                final_state_update += state_update.data
            return StateUpdate(final_state_update)

class MLPNeuron:
    """A neuron takes in a state and returns a state update. 
    The final state update of the MLP layer is just the sum of all
    the per-neuron state updates
    """
    # To convert from state to a scalar.
    # Also the column vector in the (D_EMBED, D_MLP) matrix.
    read_vector: Annotated[NDArrayFloat, D_EMBED]
    # To convert from scalar to state update.
    # Also the column vector in the (D_MLP, D_EMBED) matrix.
    write_vector: Annotated[NDArrayFloat, D_EMBED]

    def __init__(self, read_vector, write_vector):
        self.read_vector = read_vector
        self.write_vector = write_vector   
        
    def run(self, state: State) -> StateUpdate:
        scalar = np.dot(self.read_vector, state.data)
        # IRL they use GELU for the non-linearity. I just pick tanh because it's in numpy.
        scalar = np.tanh(scalar)
        return StateUpdate(self.write_vector * scalar)


def softmax(x: list[float]) -> list[float]:
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) 


def main():
    """Build a transformer with random parameter values and run it on a random input."""
    # Initialize the embedder and unembedder
    embedding_matrix = {TokenId(token_id) : np.random.rand(D_EMBED) for token_id in range(N_VOCAB)}
    token_to_state_embedder = TokenToStateEmbedder(embedding_matrix)
    state_to_token_logits_unembedder = StateToTokenLogitsUnembedder(
        {TokenId(token_id) : LogitFn(np.random.rand(D_EMBED)) for token_id in range(N_VOCAB)})

    # Initialize the residual blocks
    blocks = []
    for _ in range(N_BLOCKS):
        attention_layer = AttentionLayer([AttentionHead(
            q_projector = np.random.rand(D_ATTENTION_QUERY, D_EMBED),
            k_projector = np.random.rand(D_ATTENTION_QUERY, D_EMBED),
            v_projector = np.random.rand(D_ATTENTION, D_EMBED),
            attention_to_state_projector = np.random.rand(D_EMBED, D_ATTENTION)
        ) for _ in range(N_HEADS)])
        mlp_layer = MLPLayer([MLPNeuron(
            read_vector=np.random.rand(D_EMBED),
            write_vector=np.random.rand(D_EMBED)
        ) for _ in range(D_MLP)])
        blocks.append(Block(attention_layer, mlp_layer))

    # Create transformer
    transformer = Transformer(token_to_state_embedder, state_to_token_logits_unembedder, blocks)

    # Run transformer
    input_tokens = [TokenId(np.random.randint(0, N_VOCAB)) for i in range(N_TOKEN)]
    logits = transformer.run(input_tokens)

    # The last logit is for the next subword conditioned on the entire input context.
    print(len(logits[-1]))

if __name__ == "__main__":
    main()
