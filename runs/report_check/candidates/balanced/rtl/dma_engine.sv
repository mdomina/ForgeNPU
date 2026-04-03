module dma_engine #(
  parameter int DATA_WIDTH = 8,
  parameter int ROWS = 2,
  parameter int COLS = 2,
  parameter int DEPTH = 4,
  parameter int ADDR_WIDTH = $clog2(DEPTH),
  parameter int MAX_DIM = (ROWS > COLS) ? ROWS : COLS
) (
  input  logic clk,
  input  logic rst_n,
  input  logic dma_valid_i,
  input  logic dma_write_weights_i,
  input  logic [ADDR_WIDTH-1:0] dma_addr_i,
  input  logic signed [MAX_DIM*DATA_WIDTH-1:0] dma_payload_i,
  output logic write_activations_en_o,
  output logic [ADDR_WIDTH-1:0] activation_write_addr_o,
  output logic signed [ROWS*DATA_WIDTH-1:0] activations_write_data_o,
  output logic write_weights_en_o,
  output logic [ADDR_WIDTH-1:0] weight_write_addr_o,
  output logic signed [COLS*DATA_WIDTH-1:0] weights_write_data_o,
  output logic dma_done_o,
  output logic dma_busy_o
);
  logic pending_valid_q;
  logic pending_write_weights_q;
  logic [ADDR_WIDTH-1:0] pending_addr_q;
  logic signed [MAX_DIM*DATA_WIDTH-1:0] pending_payload_q;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      pending_valid_q <= 1'b0;
      pending_write_weights_q <= 1'b0;
      pending_addr_q <= '0;
      pending_payload_q <= '0;
    end else begin
      pending_valid_q <= dma_valid_i;
      if (dma_valid_i) begin
        pending_write_weights_q <= dma_write_weights_i;
        pending_addr_q <= dma_addr_i;
        pending_payload_q <= dma_payload_i;
      end
    end
  end

  always_comb begin
    write_activations_en_o = pending_valid_q && !pending_write_weights_q;
    activation_write_addr_o = pending_addr_q;
    activations_write_data_o = pending_payload_q[0 +: ROWS*DATA_WIDTH];

    write_weights_en_o = pending_valid_q && pending_write_weights_q;
    weight_write_addr_o = pending_addr_q;
    weights_write_data_o = pending_payload_q[0 +: COLS*DATA_WIDTH];

    dma_done_o = pending_valid_q;
    dma_busy_o = pending_valid_q || dma_valid_i;
  end
endmodule
