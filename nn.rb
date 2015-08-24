require 'matrix'
require 'pry'
require 'pry-nav'

def htan(z)
  return (Math.exp(2 * z) - 1) / (Math.exp(2 * z) + 1)
end

def htan_prime(z)
  return 1 - ((Math.exp(2 * z) - 1) / (Math.exp(2 * z) + 1) ** 2)
end

def sigmoid(z)
  return 1 / (1 + Math.exp(-z))
end

def sigmoid_prime(z)
  return Math.exp(-z) / (1 + Math.exp(-z) ** 2)
end

def sample
  return Math.sqrt(-2 * Math.log(rand(0.0..1.0))) * Math.cos(2 * Math::PI * rand(0.0..1.0))
end

#Monkey patch multiplication
class Matrix
  def *(m)
    result = Array.new(m.row_count){Array.new(column_count)}
    i = 0
    while i < m.row_count do
      j = 0
      while j < column_count do
        sum = 0
        k = 0
        while k < row_count do
          sum += self[k,j] * m[i,k]
          k += 1
        end
        result[i][j] = sum
        j += 1
      end
      i += 1
    end
    return Matrix.rows(result)
  end
end

class Nn
  def initialize(activator="sigmoid", learning_rate=0.7, iterations=10000, hidden_units=3, hidden_layers=1)
    @activate = activator
    @activate_prime = "#{activator}_prime"
    @learning_rate = learning_rate
    @iterations = iterations
    @hidden_units = hidden_units
    @hidden_layers = hidden_layers
    @weights = []
  end

  def create_matrices(data)
    ret = { input: [], output: [] }
    for i in data
      ret[:output].push(i[:output])
      ret[:input].push(i[:input])
    end
    ret[:output] = Matrix.rows(ret[:output])
    ret[:input] = Matrix.rows(ret[:input])
    return ret
  end

  #Element wise multiplication
  def multiply_elements(m1, m2)
    result = Array.new(m1.row_count){Array.new(m1.column_count)}
    i = 0
    while i<m1.row_count do
      j = 0
      while j<m1.column_count do
        result[i][j] = m1[i,j] * m2[i,j]
        j += 1
      end
      i += 1
    end
    return Matrix.rows(result)
  end

  def fsum(weight, input)
    res = {}
    res[:sum] = weight * input
    res[:result] = res[:sum].collect{|e| send(@activate, e)}
    return res
  end

  def forward(examples)
    results = []

    results.push(fsum(@weights[0], examples[:input]))

    i = 1
    while i<@hidden_layers do
      results.push( fsum(@weights[i], results[i - 1][:result]) )
      i += 1
    end

    results.push( fsum(@weights[@weights.length - 1], results[results.length - 1][:result]) )

    return results
  end

  def back(examples, results)
    error = examples[:output] - results[results.length - 1][:result]
    delta = multiply_elements(results[results.length - 1][:sum].collect{|e| send(@activate_prime, e)}, error)
    changes = (delta * results[0][:result].transpose).collect{|e| e * @learning_rate}
    @weights[@weights.length - 1] = @weights[@weights.length - 1] + changes

    i=1
    while i<@hidden_layers do
      delta = multiply_elements(@weights[@weights.length - i].transpose * delta, results[results.length - (i + 1)][:sum].collect{|e2| send(@activate_prime, e2)})
      changes = (delta * results[results.length - (i + 1)][:result].transpose).collect{|e2| e2 * @learning_rate}
      @weights[@weights.length - (i + 1)] = @weights[@weights.length - (i + 1)] + changes
      i += 1
    end

    delta = multiply_elements(@weights[1].transpose * delta, results[0][:sum].collect{|e| send(@activate_prime, e)})
    changes = (delta * examples[:input].transpose).collect{|e| e * @learning_rate}
    @weights[0] = @weights[0] + changes

    return error
  end

  def learn(examples)
    examples = create_matrices(examples)

    @weights.push(Matrix.build(examples[:input].row(0).size, @hidden_units) {sample})
    (@hidden_layers-1).times{ @weights.push(Matrix.build(@hidden_units, @hidden_units) {sample}) }
    @weights.push(Matrix.build(@hidden_units, examples[:output].row(0).size) {sample})

    @iterations.times do |i|
      puts "iteration #{i}" if i % 1000 == 0
      results = forward(examples)
      errors = back(examples, results)
    end
  end

  def predict(input)
    results = forward({ input: Matrix[input] })
    return results[results.length - 1][:result][0,0]
  end
end

net = Nn.new
net.learn([
  { input: [0, 0], output: [ 0 ] },
  { input: [0, 1], output: [ 1 ] },
  { input: [1, 0], output: [ 1 ] },
  { input: [1, 1], output: [ 0 ] }
  ])

puts net.predict([1,0])
