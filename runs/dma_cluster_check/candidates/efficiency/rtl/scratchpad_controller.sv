module scratchpad_controller #(
  parameter int DATA_WIDTH = 8,
  parameter int ROWS = 2,
  parameter int COLS = 2,
  parameter int DEPTH = 4,
  parameter int ADDR_WIDTH = $clog2(DEPTH)
) (
  input  logic clk,
  input  logic rst_n,
  input  logic write_activations_en_i,
  input  logic [ADDR_WIDTH-1:0] activation_write_addr_i,
  input  logic signed [ROWS*DATA_WIDTH-1:0] activations_write_data_i,
  input  logic write_weights_en_i,
  input  logic [ADDR_WIDTH-1:0] weight_write_addr_i,
  input  logic signed [COLS*DATA_WIDTH-1:0] weights_write_data_i,
  input  logic load_vector_en_i,
  input  logic [ADDR_WIDTH-1:0] activation_read_addr_i,
  input  logic [ADDR_WIDTH-1:0] weight_read_addr_i,
  output logic signed [ROWS*DATA_WIDTH-1:0] activations_west_o,
  output logic signed [COLS*DATA_WIDTH-1:0] weights_north_o,
  output logic vector_valid_o
);
  logic signed [ROWS*DATA_WIDTH-1:0] activation_bank [0:DEPTH-1];
  logic signed [COLS*DATA_WIDTH-1:0] weight_bank [0:DEPTH-1];
  integer bank_idx;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (bank_idx = 0; bank_idx < DEPTH; bank_idx = bank_idx + 1) begin
        activation_bank[bank_idx] <= '0;
        weight_bank[bank_idx] <= '0;
      end
    end else begin
      if (write_activations_en_i) begin
        activation_bank[activation_write_addr_i] <= activations_write_data_i;
      end
      if (write_weights_en_i) begin
        weight_bank[weight_write_addr_i] <= weights_write_data_i;
      end
    end
  end

  always_comb begin
    activations_west_o = activation_bank[activation_read_addr_i];
    weights_north_o = weight_bank[weight_read_addr_i];
    vector_valid_o = load_vector_en_i;
  end
endmodule
